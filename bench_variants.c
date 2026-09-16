/* bench_variants.c -- 回答"能不能不用 fdiv / 取倒数会不会更快"
 *
 * 沿两条正交的轴做组合:
 *   轴 1: step = range/255 怎么算      -> vdiv | vmul(1/255)
 *   轴 2: inv = 1/step 怎么算          -> vdiv | vrecpe(f32)+N*Newton | f16 倒数
 *   轴 3: 量化 (v-min)*inv 在哪个域做  -> f32 | f16
 *
 * 只替换 compress, 解压统一用 fc_decompress_neon, 保证比较公平.
 */
#include "fcompress.h"

#include <arm_neon.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static inline uint64_t xs(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

#define NPAT 7
static void gen(fc_f16 *m, uint64_t seed, int pattern)
{
    uint64_t s = seed | 1;
    static const double mags[6] = {1e-3, 1.0, 1e3, 60.0, 6e-5, 1000.0};

    for (int c = 0; c < FC_N; c++) {
        double base, range;
        switch (pattern) {
        case 0: base = 0.0; range = 1.0; break;
        case 1: base = (double)(c - 16) * 0.5; range = 1.0 + 0.25 * c; break;
        case 2: base = ((c & 1) ? -1.0 : 1.0) * mags[c % 6]; range = mags[(c / 2) % 6] * 0.5; break;
        case 3: base = 1.0; range = 0.0; break;
        case 4: base = 3.0; range = 1e-2; break;
        case 5: base = 1000.0; range = 0.5; break;  /* range = 1 ulp(min) */
        case 6: base = 0.0; range = 5e-4; break;    /* step 落在 f16 次正规区 */
        default: base = 0.0; range = 1.0; break;
        }
        for (int r = 0; r < FC_N; r++) {
            double u = (double)((xs(&s) >> 11) & 0xFFFFFFu) / (double)(1u << 24);
            double v = base + range * u;
            if (v > 65504.0) v = 65504.0;
            if (v < -65504.0) v = -65504.0;
            m[r * FC_N + c] = (fc_f16)v;
        }
    }
}

/* ================= 共享: pass 1 求 32 列 min/max ================= */
static inline void minmax32(const float16_t *p, float16x8_t mn[4], float16x8_t mx[4])
{
    mn[0] = vld1q_f16(p + 0);
    mn[1] = vld1q_f16(p + 8);
    mn[2] = vld1q_f16(p + 16);
    mn[3] = vld1q_f16(p + 24);
    for (int g = 0; g < 4; g++) mx[g] = mn[g];
    for (int r = 1; r < FC_N; r++) {
        const float16_t *row = p + r * FC_N;
        for (int g = 0; g < 4; g++) {
            float16x8_t v = vld1q_f16(row + g * 8);
            mn[g] = vminq_f16(mn[g], v);
            mx[g] = vmaxq_f16(mx[g], v);
        }
    }
}

static inline void store_minstep(fc_block *out, int g, float32x4_t lo_mn, float32x4_t hi_mn,
                                 float16x8_t step16)
{
    vst1q_f16((float16_t *)out->min + g * 8,
              vcombine_f16(vcvt_f16_f32(lo_mn), vcvt_f16_f32(hi_mn)));
    vst1q_f16((float16_t *)out->step + g * 8, step16);
}

/* ================= 轴 1: step = range/255 ================= */
static inline float16x8_t step_by_div(float32x4_t lo_r, float32x4_t hi_r)
{
    const float32x4_t c255 = vdupq_n_f32(255.0f);
    return vcombine_f16(vcvt_f16_f32(vdivq_f32(lo_r, c255)),
                        vcvt_f16_f32(vdivq_f32(hi_r, c255)));
}
static inline float16x8_t step_by_mul(float32x4_t lo_r, float32x4_t hi_r)
{
    const float32x4_t ci = vdupq_n_f32(1.0f / 255.0f);
    return vcombine_f16(vcvt_f16_f32(vmulq_f32(lo_r, ci)), vcvt_f16_f32(vmulq_f32(hi_r, ci)));
}

/* ================= 轴 2: inv = 1/step16 ================= */
static inline float32x4_t rcp_f32_1(float32x4_t x)
{
    float32x4_t e = vrecpeq_f32(x);
    return vmulq_f32(vrecpsq_f32(x, e), e); /* 1 次 Newton: ~2^-16 */
}
static inline float32x4_t rcp_f32_2(float32x4_t x)
{
    float32x4_t e = rcp_f32_1(x);
    return vmulq_f32(vrecpsq_f32(x, e), e); /* 2 次 Newton: ~2^-23 */
}

/* ================= 参数打包 ================= */
typedef struct {
    float32x4_t lo_mn[4], hi_mn[4], lo_inv[4], hi_inv[4];
} f32p;
typedef struct {
    float16x8_t mn[4], inv[4];
} f16p;

static const float32x4_t ZERO32 = {0, 0, 0, 0};
static const float16x8_t ZERO16 = {0, 0, 0, 0, 0, 0, 0, 0};

/* ================= 轴 3: 量化 ================= */
static void quant_f32(const float16_t *p, fc_block *out, const f32p *q)
{
    for (int r = 0; r < FC_N; r++) {
        const float16_t *row = p + r * FC_N;
        uint8_t *dst = out->q + r * FC_N;
        for (int g = 0; g < 4; g++) {
            float16x8_t v = vld1q_f16(row + g * 8);
            float32x4_t vl = vcvt_f32_f16(vget_low_f16(v));
            float32x4_t vh = vcvt_high_f32_f16(v);
            uint32x4_t ul = vcvtnq_u32_f32(vmulq_f32(vsubq_f32(vl, q->lo_mn[g]), q->lo_inv[g]));
            uint32x4_t uh = vcvtnq_u32_f32(vmulq_f32(vsubq_f32(vh, q->hi_mn[g]), q->hi_inv[g]));
            vst1_u8(dst + g * 8, vqmovn_u16(vcombine_u16(vqmovn_u32(ul), vqmovn_u32(uh))));
        }
    }
}

/* f16 域量化: 6 条指令 / 8 元素; mode: 0=就近偶数 1=截断 */
static void quant_f16(const float16_t *p, fc_block *out, const f16p *q, int mode)
{
    for (int r = 0; r < FC_N; r++) {
        const float16_t *row = p + r * FC_N;
        uint8_t *dst = out->q + r * FC_N;
        for (int g = 0; g < 4; g++) {
            float16x8_t v = vld1q_f16(row + g * 8);
            float16x8_t t = vmulq_f16(vsubq_f16(v, q->mn[g]), q->inv[g]);
            uint16x8_t u = mode ? vcvtq_u16_f16(t) : vcvtnq_u16_f16(t);
            vst1_u8(dst + g * 8, vqmovn_u16(u));
        }
    }
}

/* ================= 7 个变体 ================= */
#define MID_COMMON(g, lo_mn, hi_mn, lo_r, hi_r)                        \
    float32x4_t lo_mn = vcvt_f32_f16(vget_low_f16(mn[g]));             \
    float32x4_t hi_mn = vcvt_high_f32_f16(mn[g]);                      \
    float32x4_t lo_r = vsubq_f32(vcvt_f32_f16(vget_low_f16(mx[g])), lo_mn); \
    float32x4_t hi_r = vsubq_f32(vcvt_high_f32_f16(mx[g]), hi_mn)

/* A: div + div (当前实现) */
static void comp_A(const fc_f16 *restrict in, fc_block *restrict out)
{
    const float16_t *p = (const float16_t *)in;
    float16x8_t mn[4], mx[4];
    minmax32(p, mn, mx);
    f32p q;
    for (int g = 0; g < 4; g++) {
        MID_COMMON(g, lo_mn, hi_mn, lo_r, hi_r);
        float16x8_t st = step_by_div(lo_r, hi_r);
        store_minstep(out, g, lo_mn, hi_mn, st);
        q.lo_mn[g] = lo_mn;
        q.hi_mn[g] = hi_mn;
        float32x4_t sl = vcvt_f32_f16(vget_low_f16(st)), sh = vcvt_high_f32_f16(st);
        q.lo_inv[g] = vbslq_f32(vceqq_f32(sl, ZERO32), ZERO32, vdivq_f32(vdupq_n_f32(1.0f), sl));
        q.hi_inv[g] = vbslq_f32(vceqq_f32(sh, ZERO32), ZERO32, vdivq_f32(vdupq_n_f32(1.0f), sh));
    }
    quant_f32(p, out, &q);
}

/* B: mul(1/255) + div  —— 只去掉"除以常数"那次除法 */
static void comp_B(const fc_f16 *restrict in, fc_block *restrict out)
{
    const float16_t *p = (const float16_t *)in;
    float16x8_t mn[4], mx[4];
    minmax32(p, mn, mx);
    f32p q;
    for (int g = 0; g < 4; g++) {
        MID_COMMON(g, lo_mn, hi_mn, lo_r, hi_r);
        float16x8_t st = step_by_mul(lo_r, hi_r);
        store_minstep(out, g, lo_mn, hi_mn, st);
        q.lo_mn[g] = lo_mn;
        q.hi_mn[g] = hi_mn;
        float32x4_t sl = vcvt_f32_f16(vget_low_f16(st)), sh = vcvt_high_f32_f16(st);
        q.lo_inv[g] = vbslq_f32(vceqq_f32(sl, ZERO32), ZERO32, vdivq_f32(vdupq_n_f32(1.0f), sl));
        q.hi_inv[g] = vbslq_f32(vceqq_f32(sh, ZERO32), ZERO32, vdivq_f32(vdupq_n_f32(1.0f), sh));
    }
    quant_f32(p, out, &q);
}

/* C: mul + f32 倒数 1 次 Newton */
static void comp_C(const fc_f16 *restrict in, fc_block *restrict out)
{
    const float16_t *p = (const float16_t *)in;
    float16x8_t mn[4], mx[4];
    minmax32(p, mn, mx);
    f32p q;
    for (int g = 0; g < 4; g++) {
        MID_COMMON(g, lo_mn, hi_mn, lo_r, hi_r);
        float16x8_t st = step_by_mul(lo_r, hi_r);
        store_minstep(out, g, lo_mn, hi_mn, st);
        q.lo_mn[g] = lo_mn;
        q.hi_mn[g] = hi_mn;
        q.lo_inv[g] = vbslq_f32(vceqq_f32(vcvt_f32_f16(vget_low_f16(st)), ZERO32), ZERO32,
                                rcp_f32_1(vcvt_f32_f16(vget_low_f16(st))));
        q.hi_inv[g] = vbslq_f32(vceqq_f32(vcvt_high_f32_f16(st), ZERO32), ZERO32,
                                rcp_f32_1(vcvt_high_f32_f16(st)));
    }
    quant_f32(p, out, &q);
}

/* D: mul + f32 倒数 2 次 Newton (上一轮的实验版) */
static void comp_D(const fc_f16 *restrict in, fc_block *restrict out)
{
    const float16_t *p = (const float16_t *)in;
    float16x8_t mn[4], mx[4];
    minmax32(p, mn, mx);
    f32p q;
    for (int g = 0; g < 4; g++) {
        MID_COMMON(g, lo_mn, hi_mn, lo_r, hi_r);
        float16x8_t st = step_by_mul(lo_r, hi_r);
        store_minstep(out, g, lo_mn, hi_mn, st);
        q.lo_mn[g] = lo_mn;
        q.hi_mn[g] = hi_mn;
        q.lo_inv[g] = vbslq_f32(vceqq_f32(vcvt_f32_f16(vget_low_f16(st)), ZERO32), ZERO32,
                                rcp_f32_2(vcvt_f32_f16(vget_low_f16(st))));
        q.hi_inv[g] = vbslq_f32(vceqq_f32(vcvt_high_f32_f16(st), ZERO32), ZERO32,
                                rcp_f32_2(vcvt_high_f32_f16(st)));
    }
    quant_f32(p, out, &q);
}

/* E: mul + f16 倒数(1 Newton) + f16 域量化 + FCVTNU.8H  —— 全 f16 路线 */
static void comp_E(const fc_f16 *restrict in, fc_block *restrict out)
{
    const float16_t *p = (const float16_t *)in;
    float16x8_t mn[4], mx[4];
    minmax32(p, mn, mx);
    f16p q;
    for (int g = 0; g < 4; g++) {
        MID_COMMON(g, lo_mn, hi_mn, lo_r, hi_r);
        float16x8_t st = step_by_mul(lo_r, hi_r);
        store_minstep(out, g, lo_mn, hi_mn, st);
        float16x8_t e = vrecpeq_f16(st);
        e = vmulq_f16(vrecpsq_f16(st, e), e);
        q.mn[g] = mn[g];
        q.inv[g] = vbslq_f16(vceqq_f16(st, ZERO16), ZERO16, e);
    }
    quant_f16(p, out, &q, 0);
}

/* E0: 同 E, 但 f16 倒数不做 Newton (FRECPE 只有 ~8bit) */
static void comp_E0(const fc_f16 *restrict in, fc_block *restrict out)
{
    const float16_t *p = (const float16_t *)in;
    float16x8_t mn[4], mx[4];
    minmax32(p, mn, mx);
    f16p q;
    for (int g = 0; g < 4; g++) {
        MID_COMMON(g, lo_mn, hi_mn, lo_r, hi_r);
        float16x8_t st = step_by_mul(lo_r, hi_r);
        store_minstep(out, g, lo_mn, hi_mn, st);
        q.mn[g] = mn[g];
        q.inv[g] = vbslq_f16(vceqq_f16(st, ZERO16), ZERO16, vrecpeq_f16(st));
    }
    quant_f16(p, out, &q, 0);
}

/* ET: 同 E, 但用截断 FCVTZU 而不是就近偶数 */
static void comp_ET(const fc_f16 *restrict in, fc_block *restrict out)
{
    const float16_t *p = (const float16_t *)in;
    float16x8_t mn[4], mx[4];
    minmax32(p, mn, mx);
    f16p q;
    for (int g = 0; g < 4; g++) {
        MID_COMMON(g, lo_mn, hi_mn, lo_r, hi_r);
        float16x8_t st = step_by_mul(lo_r, hi_r);
        store_minstep(out, g, lo_mn, hi_mn, st);
        float16x8_t e = vrecpeq_f16(st);
        e = vmulq_f16(vrecpsq_f16(st, e), e);
        q.mn[g] = mn[g];
        q.inv[g] = vbslq_f16(vceqq_f16(st, ZERO16), ZERO16, e);
    }
    quant_f16(p, out, &q, 1);
}

/* EG == 库里已固化的 fc_compress_neon: 全 f16 快路径 + step 过小时回退 f32.
 * 这里直接调用库函数, 保证表格里的数字就是实际交付代码的数字. */
static void comp_EG(const fc_f16 *restrict in, fc_block *restrict out)
{
    fc_compress_neon(in, out);
}

/* ================= 驱动器 ================= */
typedef void (*comp_fn)(const fc_f16 *restrict, fc_block *restrict);
static const struct {
    const char *tag;
    const char *name;
    comp_fn fn;
} VARS[] = {
    {"A", "div + div(1/step)        f32-quant", comp_A},
    {"B", "mul + div(1/step)        f32-quant", comp_B},
    {"C", "mul + recpe/1*Newton     f32-quant", comp_C},
    {"D", "mul + recpe/2*Newton     f32-quant", comp_D},
    {"E", "mul + frecpe/1*Newton    f16-quant", comp_E},
    {"E0", "mul + frecpe/no Newton   f16-quant", comp_E0},
    {"ET", "mul + frecpe/1*Newton    f16-quant/trunc", comp_ET},
    {"FAST", "库 fc_compress_neon     f16-quant+守卫", comp_EG},
    {"HP", "库 fc_compress_neon_hp  f32-quant (基线)", fc_compress_neon_hp},
};
#define NVAR ((int)(sizeof(VARS) / sizeof(VARS[0])))

static double snr_db(const fc_f16 *ref, const fc_f16 *got)
{
    double se = 0.0, sr = 0.0;
    for (int i = 0; i < FC_ELEMS; i++) {
        double a = (double)ref[i], b = (double)got[i];
        se += (a - b) * (a - b);
        sr += a * a;
    }
    if (se == 0.0) return INFINITY;
    if (sr == 0.0) return 0.0;
    return 10.0 * log10(sr / se);
}

static double recon_snr(const fc_f16 *in, const fc_block *b)
{
    fc_f16 out[FC_ELEMS];
    fc_decompress_neon(b, out);
    return snr_db(in, out);
}

int main(void)
{
    int fails = 0;
    printf("### 1) 正确性: 各变体 vs 基线 A (7 种数据模式)\n\n");
    printf("  %-3s %-38s %8s %8s %8s %9s\n", "变体", "", "Σ|Δq|", "max|Δq|", "Δstep列", "最差ΔSNR");
    for (int v = 0; v < NVAR; v++) {
        long dq = 0;
        int maxdq = 0, dstep = 0;
        double worst = 1e30;
        for (int pat = 0; pat < NPAT; pat++) {
            fc_f16 in[FC_ELEMS];
            fc_block ba, bv;
            gen(in, 0x1234567 + pat, pat);
            comp_A(in, &ba);
            VARS[v].fn(in, &bv);
            for (int i = 0; i < FC_ELEMS; i++) {
                int d = (int)bv.q[i] - (int)ba.q[i];
                if (d) dq += (d < 0 ? -d : d);
                if (abs(d) > maxdq) maxdq = abs(d);
            }
            for (int c = 0; c < FC_N; c++)
                if (memcmp(&bv.step[c], &ba.step[c], 2) != 0) dstep++;
            double sa = recon_snr(in, &ba), sb = recon_snr(in, &bv);
            double d = (isinf(sa) && isinf(sb)) ? 0.0 : sb - sa;
            if (d < worst) worst = d;
        }
        printf("  %-3s %-38s %8ld %8d %8d %9.2f\n", VARS[v].tag, VARS[v].name, dq, maxdq, dstep,
               worst);
        const char *tg = VARS[v].tag; /* E/E0/ET 是"故意做错"的对照组, 预期崩坏 */
        if (maxdq > 1 && strcmp(tg, "E") && strcmp(tg, "E0") && strcmp(tg, "ET")) fails++;
    }

    printf("\n### 2) 精度细节: 关键模式下的重建 SNR (dB)\n\n");
    printf("  %-6s", "模式");
    for (int v = 0; v < NVAR; v++) printf(" %7s", VARS[v].tag);
    printf("\n");
    for (int pat = 0; pat < NPAT; pat++) {
        fc_f16 in[FC_ELEMS];
        gen(in, 0x1234567 + pat, pat);
        printf("  pat%-3d", pat);
        for (int v = 0; v < NVAR; v++) {
            fc_block b;
            VARS[v].fn(in, &b);
            double s = recon_snr(in, &b);
            if (isinf(s)) printf("     inf");
            else printf(" %7.2f", s);
        }
        printf("\n");
    }

    printf("\n### 3) 速度: L2 常驻 512 块 x 4000 轮\n\n");
    {
        const int nres = 512, reps = 4000;
        fc_f16 *arr = aligned_alloc(64, (size_t)nres * 2048);
        fc_block *cb = aligned_alloc(64, (size_t)nres * sizeof(fc_block));
        for (int i = 0; i < nres; i++) gen(arr + (size_t)i * FC_ELEMS, 0x55 + i, 1);

        for (int v = 0; v < NVAR; v++) {
            for (int i = 0; i < nres; i++) VARS[v].fn(arr + (size_t)i * FC_ELEMS, &cb[i]);
            double t0 = now_s();
            for (int k = 0; k < reps; k++)
                for (int i = 0; i < nres; i++) VARS[v].fn(arr + (size_t)i * FC_ELEMS, &cb[i]);
            double t1 = now_s();
            long n = (long)reps * nres;
            printf("  %-3s %-38s %7.2f ns/块  %6.2f GB/s\n", VARS[v].tag, VARS[v].name, (t1 - t0) / n * 1e9,
                   4096.0 * n / (t1 - t0) / 1e9);
        }
        uint64_t sink = cb[0].q[0];
        printf("  (sink=%llu)\n", (unsigned long long)sink);
        free(arr);
        free(cb);
    }
    return fails != 0;
}
