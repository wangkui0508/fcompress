/* fcompress_ref.c -- 标量参考实现, 运算顺序刻意与 NEON 版保持一致,
 * 用于对拍量化值 q / min / step 是否 bit-exact。 */
#include "fcompress.h"

#include <math.h>

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
