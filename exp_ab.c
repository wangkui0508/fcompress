/* exp_ab.c -- 同进程 A/B: 原始 fc_minmax/守卫 (OLD) vs 当前库 (NEW, 调库) */
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
    uint64_t x = *s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; *s = x; return x;
}
static void gen(fc_f16 *m, uint64_t seed, int pattern)
{
    uint64_t s = seed | 1;
    double base, range;
    switch (pattern) {
    case 0: base = 0.0; range = 1.0; break;
    case 1: base = -4.0; range = 9.0; break;
    case 3: base = 1.0; range = 0.0; break;
    case 4: base = 3.0; range = 1e-2; break;
    case 5: base = 0.0; range = 5e-4; break;
    case 6: base = -32752.0; range = 65504.0; break;
    case 7: base = -60000.0; range = 120000.0; break;
    case 8: base = -65504.0; range = 131008.0; break;
    default: base = 0.0; range = 1.0; break;
    }
    for (int r = 0; r < FC_ELEMS; r++) {
        double u = (double)((xs(&s) >> 11) & 0xFFFFFFu) / (double)(1u << 24);
        double v = base + range * u;
        if (v > 65504.0) v = 65504.0;
        if (v < -65504.0) v = -65504.0;
        m[r] = (fc_f16)v;
    }
}

#define FC_FAST_STEP_MIN 2.0e-5f
#define FC_F16_MAX 65504.0f

/* ===== 原始实现的 minmax (4 组串行链) ===== */
static inline void minmax_old(const float16_t *p, float16x8_t mn[4], float16x8_t mx[4])
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

/* ===== 原始实现的完整 compress (逐条复刻原 fcompress_neon.c) ===== */
static inline float16x8_t fc_step_store_old(const float16x8_t mn[4], const float16x8_t mx[4],
                                     fc_block *out, int g, float32x4_t *lo_mn, float32x4_t *hi_mn)
{
    const float32x4_t c255 = vdupq_n_f32(255.0f);
    float32x4_t lmn = vcvt_f32_f16(vget_low_f16(mn[g]));
    float32x4_t hmn = vcvt_high_f32_f16(mn[g]);
    float32x4_t lr = vsubq_f32(vcvt_f32_f16(vget_low_f16(mx[g])), lmn);
    float32x4_t hr = vsubq_f32(vcvt_high_f32_f16(mx[g]), hmn);
    float16x8_t step16 =
        vcombine_f16(vcvt_f16_f32(vdivq_f32(lr, c255)), vcvt_f16_f32(vdivq_f32(hr, c255)));
    vst1q_f16((float16_t *)out->min + g * 8, vcombine_f16(vcvt_f16_f32(lmn), vcvt_f16_f32(hmn)));
    vst1q_f16((float16_t *)out->step + g * 8, step16);
    *lo_mn = lmn;
    *hi_mn = hmn;
    return step16;
}
static void quant_f32_old(const float16_t *p, fc_block *out, const float32x4_t lo_mn[4],
                          const float32x4_t hi_mn[4], const float32x4_t lo_inv[4],
                          const float32x4_t hi_inv[4])
{
    for (int r = 0; r < FC_N; r++) {
        const float16_t *row = p + r * FC_N;
        uint8_t *dst = out->q + r * FC_N;
        for (int g = 0; g < 4; g++) {
            float16x8_t v = vld1q_f16(row + g * 8);
            float32x4_t vl = vcvt_f32_f16(vget_low_f16(v));
            float32x4_t vh = vcvt_high_f32_f16(v);
            uint32x4_t ul = vcvtnq_u32_f32(vmulq_f32(vsubq_f32(vl, lo_mn[g]), lo_inv[g]));
            uint32x4_t uh = vcvtnq_u32_f32(vmulq_f32(vsubq_f32(vh, hi_mn[g]), hi_inv[g]));
            vst1_u8(dst + g * 8, vqmovn_u16(vcombine_u16(vqmovn_u32(ul), vqmovn_u32(uh))));
        }
    }
}

static void comp_OLD(const fc_f16 *restrict in, fc_block *restrict out)
{
    const float16_t *p = (const float16_t *)in;
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    float16x8_t mn[4], mx[4], step16[4];
    float32x4_t lo_mn[4], hi_mn[4], lo_inv[4], hi_inv[4];

    minmax_old(p, mn, mx);
    for (int g = 0; g < 4; g++)
        step16[g] = fc_step_store_old(mn, mx, out, g, &lo_mn[g], &hi_mn[g]);

    const float16x8_t big = vdupq_n_f16(FC_F16_MAX);
    const float16x8_t zero16 = vdupq_n_f16(0.0f);
    float16x8_t acc = big, rng = zero16;
    for (int g = 0; g < 4; g++) {
        acc = vminq_f16(acc, vbslq_f16(vceqq_f16(step16[g], zero16), big, step16[g]));
        rng = vmaxq_f16(rng, vsubq_f16(mx[g], mn[g]));
    }

    if ((float)vminvq_f16(acc) < FC_FAST_STEP_MIN || (float)vmaxvq_f16(rng) > FC_F16_MAX) {
        for (int g = 0; g < 4; g++) {
            float32x4_t sl = vcvt_f32_f16(vget_low_f16(step16[g]));
            float32x4_t sh = vcvt_high_f32_f16(step16[g]);
            lo_inv[g] = vbslq_f32(vceqq_f32(sl, zero), zero, vdivq_f32(one, sl));
            hi_inv[g] = vbslq_f32(vceqq_f32(sh, zero), zero, vdivq_f32(one, sh));
        }
        quant_f32_old(p, out, lo_mn, hi_mn, lo_inv, hi_inv);
        return;
    }
    float16x8_t inv16[4];
    for (int g = 0; g < 4; g++) {
        float16x8_t e = vrecpeq_f16(step16[g]);
        e = vmulq_f16(vrecpsq_f16(step16[g], e), e);
        inv16[g] = vbslq_f16(vceqq_f16(step16[g], zero16), zero16, e);
    }
    for (int r = 0; r < FC_N; r++) {
        const float16_t *row = p + r * FC_N;
        uint8_t *dst = out->q + r * FC_N;
        for (int g = 0; g < 4; g++) {
            float16x8_t v = vld1q_f16(row + g * 8);
            float16x8_t t = vmulq_f16(vsubq_f16(v, mn[g]), inv16[g]);
            vst1_u8(dst + g * 8, vqmovn_u16(vcvtnq_u16_f16(t)));
        }
    }
}

static void comp_NEW(const fc_f16 *restrict in, fc_block *restrict out)
{
    fc_compress_neon(in, out);
}

int main(void)
{
    /* 正确性: OLD == NEW 逐位 */
    int fail = 0;
    for (int pat = 0; pat <= 8; pat++) {
        fc_f16 in[FC_ELEMS];
        static fc_block b0, b1;
        gen(in, 0x1234567 + pat, pat);
        comp_OLD(in, &b0);
        comp_NEW(in, &b1);
        if (memcmp(&b0, &b1, sizeof b0) != 0) {
            printf("pattern %d: memcmp DIFF\n", pat);
            fail = 1;
        }
    }
    printf("OLD vs NEW: %s\n", fail ? "FAIL" : "bit-exact 全 9 pattern");

    const int nres = 512, reps = 4000;
    fc_f16 *arr = aligned_alloc(64, (size_t)nres * 2048);
    fc_block *cb = aligned_alloc(64, (size_t)nres * sizeof(fc_block));
    for (int i = 0; i < nres; i++) gen(arr + (size_t)i * FC_ELEMS, 0x55 + i, 1);
    for (int i = 0; i < nres; i++) comp_NEW(arr + (size_t)i * FC_ELEMS, &cb[i]);

    /* 交替跑 10 轮, 每轮先 OLD 后 NEW 再 NEW 后 OLD 平衡场序 */
    double t_old = 0, t_new = 0;
    for (int round = 0; round < 10; round++) {
        double t0 = now_s();
        for (int k = 0; k < reps; k++)
            for (int i = 0; i < nres; i++) comp_OLD(arr + (size_t)i * FC_ELEMS, &cb[i]);
        double t1 = now_s();
        for (int k = 0; k < reps; k++)
            for (int i = 0; i < nres; i++) comp_NEW(arr + (size_t)i * FC_ELEMS, &cb[i]);
        double t2 = now_s();
        for (int k = 0; k < reps; k++)
            for (int i = 0; i < nres; i++) comp_NEW(arr + (size_t)i * FC_ELEMS, &cb[i]);
        double t3 = now_s();
        for (int k = 0; k < reps; k++)
            for (int i = 0; i < nres; i++) comp_OLD(arr + (size_t)i * FC_ELEMS, &cb[i]);
        double t4 = now_s();
        long n = (long)reps * nres;
        printf("  round %d: OLD %5.2f/%5.2f ns  NEW %5.2f/%5.2f ns\n", round, (t1 - t0) / n * 1e9,
               (t4 - t3) / n * 1e9, (t2 - t1) / n * 1e9, (t3 - t2) / n * 1e9);
        t_old += (t1 - t0) + (t4 - t3);
        t_new += (t2 - t1) + (t3 - t2);
    }
    long n = (long)reps * nres;
    printf("  合计  : OLD %6.2f   NEW %6.2f ns/块  (%+.1f%%)\n", t_old / n * 1e9, t_new / n * 1e9,
           (t_new - t_old) / t_old * 100.0);
    printf("  (sink=%u)\n", cb[0].q[0]);
    free(arr);
    free(cb);
    return fail;
}
