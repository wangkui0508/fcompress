/* fcompress_ref.c -- 标量参考实现, 运算顺序刻意与 NEON 版保持一致,
 * 用于对拍量化值 q / min / step 是否 bit-exact。 */
#include "fcompress.h"

#include <math.h>
#include <string.h>

void fc_compress_ref(const fc_f16 *restrict in, fc_block *restrict out)
{
    for (int c = 0; c < FC_N; c++) {
        float mn = (float)in[c];
        float mx = mn;
        for (int r = 1; r < FC_N; r++) {
            float v = (float)in[r * FC_N + c];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }

        /* 与 NEON 完全相同的顺序: f32 减法 -> f32 除法 -> 窄化 f16 */
        float step32 = (mx - mn) / 255.0f;
        fc_f16 step16 = (fc_f16)step32;
        float inv = (step16 == (fc_f16)0.0f) ? 0.0f : 1.0f / (float)step16;

        out->min[c] = (fc_f16)mn;
        out->step[c] = step16;

        for (int r = 0; r < FC_N; r++) {
            float t = ((float)in[r * FC_N + c] - mn) * inv;
            int qi;
            if (!(t > 0.0f)) {          /* 含 t<=0 与 NaN */
                qi = 0;
            } else if (t >= 255.0f) {
                qi = 255;
            } else {
                qi = (int)lrintf(t);    /* 就近偶数, 同 FCVTNU */
            }
            out->q[r * FC_N + c] = (uint8_t)qi;
        }
    }
}

/* 高质量路径的标量参考: 逐列从同样的候选集里挑真实 SSE 最小的网格.
 * 运算顺序刻意与 fc_compress_neon_hq 一致, 用于 bit-exact 对拍:
 *   - 1/step 用 f32 除法再窄化 (不用 FRECPE, 否则不可复现)
 *   - min' = f16(f32(min) + Δ*f32(step)): 乘法与加法分开算, 不融合
 *   - t = f16(bias + v*inv) 用 double 中间量模拟"单次舍入"的 FMA
 *   - 平方累加量是 (v-r)*inv, 每步都舍回 f16 (与 NEON 的 f16 累加一致) */
void fc_compress_ref_hq(const fc_f16 *restrict in, fc_block *restrict out)
{
    float mn[FC_N], mx[FC_N];
    fc_f16 step0[FC_N], bias0[FC_N], inv0[FC_N];

    /* ---- 阶段 1: 每列 min/max/step0, 并算块级守卫 (与 NEON 一样是块级) ---- */
    float acc = 65504.0f, rng = 0.0f;
    for (int c = 0; c < FC_N; c++) {
        mn[c] = (float)in[c];
        mx[c] = mn[c];
        for (int r = 1; r < FC_N; r++) {
            float v = (float)in[r * FC_N + c];
            if (v < mn[c]) mn[c] = v;
            if (v > mx[c]) mx[c] = v;
        }
        step0[c] = (fc_f16)((mx[c] - mn[c]) / 255.0f);
        float s = ((float)step0[c] == 0.0f) ? 65504.0f : (float)step0[c];
        if (s < acc) acc = s;
        float rr = (float)(fc_f16)((float)mx[c] - (float)mn[c]); /* f16 域减法 */
        if (rr > rng) rng = rr;
        float iv = ((float)step0[c] == 0.0f) ? 0.0f : 1.0f / (float)step0[c];
        inv0[c] = (fc_f16)iv;
        bias0[c] = (fc_f16)(-((float)mn[c] * (float)inv0[c])); /* f16 乘法: 乘积精确, 单次舍入 */
    }
    int guard12 = (acc < 2.0e-5f) || (rng > 65504.0f);
    float mag = 0.0f;
    for (int c = 0; c < FC_N; c++) {
        float m = (float)fabs((double)bias0[c]);
        if (m > mag) mag = m;
    }
    (void)mag; /* 守卫 3 只影响"快路径用不用 FMA"; HQ 全程 sub+mul, 不受它约束 */

    if (guard12) { /* 与 NEON 同: 整块退回 f32 域量化 (fc_quant_f32) */
        for (int c = 0; c < FC_N; c++) {
            out->min[c] = (fc_f16)mn[c];
            out->step[c] = step0[c];
            float inv = ((float)step0[c] == 0.0f) ? 0.0f : 1.0f / (float)step0[c];
            for (int r = 0; r < FC_N; r++) {
                float t = ((float)in[r * FC_N + c] - mn[c]) * inv; /* f32 域 */
                int qi = (!(t > 0.0f)) ? 0 : (t >= 255.0f ? 255 : (int)lrintf(t));
                out->q[r * FC_N + c] = (uint8_t)qi;
            }
        }
        return;
    }
    /* ---- 阶段 2: 逐列搜候选网格 ---- */
    for (int c = 0; c < FC_N; c++) {
        double best = 1e300;
        fc_f16 bmn = (fc_f16)mn[c], bst = step0[c], binv = inv0[c];

        for (int si = 0; si < FC_HQ_NSTEP; si++) {
            fc_f16 st = step0[c];
            {
                uint16_t bits;
                memcpy(&bits, &step0[c], 2);
                bits = (uint16_t)(bits + (si - FC_HQ_NSTEP / 2));
                memcpy(&st, &bits, 2);
            }
            fc_f16 inv = (fc_f16)(((float)st == 0.0f) ? 0.0f : 1.0f / (float)st);

            for (int j = 0; j < FC_HQ_NOFF; j++) {
                float delta = -1.0f + 2.0f * (float)j / (float)(FC_HQ_NOFF - 1);
                float t = delta * (float)st;    /* 先乘 */
                fc_f16 mp = (fc_f16)(mn[c] + t); /* 后加 (不融合, 与 NEON 的 vmul+vadd 一致) */
                (void)inv; /* 求值用 sub+mul, 不需要 bias; 但候选顺序不变 */
                fc_f16 accs = (fc_f16)0.0f;
                for (int r = 0; r < FC_N; r++) {
                    double vd = (double)in[r * FC_N + c];
                    /* sub+mul 求值: 减法/乘法各舍一次, 与 FSUB.8H+FMUL.8H 一致 */
                    fc_f16 dv = (fc_f16)(vd - (double)mp);
                    fc_f16 tf = (fc_f16)((double)dv * (double)inv);
                    int qi;
                    if (!((float)tf > 0.0f)) qi = 0;
                    else if ((float)tf >= 255.0f) qi = 255;
                    else qi = (int)llrint((double)tf);
                    fc_f16 rr = (fc_f16)((double)mp + (double)qi * (double)st); /* min + q*step */
                    fc_f16 e = (fc_f16)(vd - (double)rr);                       /* Sterbenz: 精确 */
                    fc_f16 es = (fc_f16)((double)e * (double)inv);              /* 归一化残差 */
                    accs = (fc_f16)((double)accs + (double)es * (double)es);
                }
                if ((double)accs < best) { /* 严格小于: 平局取先者, 与 vcltq 一致 */
                    best = (double)accs;
                    bmn = mp; bst = st; binv = inv;
                }
            }
        }

        out->min[c] = bmn;
        out->step[c] = bst;
        for (int r = 0; r < FC_N; r++) {
            double vd = (double)in[r * FC_N + c];
            fc_f16 tf;
            { /* 与 NEON 同: 最终量化一律 sub+mul, 与搜索求值同形式 */
                fc_f16 dv = (fc_f16)(vd - (double)bmn);
                tf = (fc_f16)((double)dv * (double)binv);
            }
            int qi;
            if (!((float)tf > 0.0f)) qi = 0;
            else if ((float)tf >= 255.0f) qi = 255;
            else qi = (int)llrint((double)tf);
            out->q[r * FC_N + c] = (uint8_t)qi;
        }
    }
}

void fc_decompress_ref(const fc_block *restrict in, fc_f16 *restrict out)
{
    for (int c = 0; c < FC_N; c++) {
        /* 用 double 累加 = 只做一次舍入, 等价于 NEON 的 FMA 语义 */
        double mn = (double)in->min[c];
        double st = (double)in->step[c];
        for (int r = 0; r < FC_N; r++) {
            out[r * FC_N + c] = (fc_f16)(mn + st * (double)in->q[r * FC_N + c]);
        }
    }
}
