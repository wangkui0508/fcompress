/* exp_opt.c -- 快路径压缩的 A/B: 优化前(OLD) vs 当前库(NEW) + 收益归因
 *
 * 变体:
 *   OLD   优化前的实现 (单行扫描 + f16 sub/mul 量化 + 4 次 8B store), 逐条复刻
 *   NEW   当前库 fc_compress_neon (4 行展开 + 整行 32B store + FMA 量化)
 *   U4    只加 4 行展开与 32B store, 量化仍是 sub/mul  (守卫 3 命中时走的就是它)
 *   FMA   只把量化换成 FMA, 不做行展开
 *   HP    fc_compress_neon_hp, f32 量化 (守卫 1/2 命中时的代价)
 *   P1    只做 pass1 (min/max + step 落盘)  —— 诊断
 *   P2    只做 pass2 (load + 量化 + store, mn/inv 取常数) —— 诊断
 *
 * 两类数据:
 *   mode 0  居中数据, |min| <= 0.5*range   (守卫 3 不命中, 走 FMA)
 *   mode 1  大 DC 数据, |min| 可达 8*range (守卫 3 命中, 走 sub/mul)
 */
#include "fcompress.h"

#include <arm_neon.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*comp_fn)(const fc_f16 *restrict, fc_block *restrict);

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
static void gen(fc_f16 *m, uint64_t seed, int mode)
{
    uint64_t s = seed | 1;
    for (int c = 0; c < FC_N; c++) {
        double base, range;
        if (mode == 1) { /* 仓库自带 bench 的 pattern 1 */
            base = (double)(c - 16) * 0.5;
            range = 1.0 + 0.25 * c;
        } else {
            range = 0.5 + (double)((xs(&s) >> 11) & 0xFFFFFFu) / (double)(1u << 24);
            base = 0.5 * range * ((double)((xs(&s) >> 11) & 0xFFFFFFu) / (double)(1u << 24));
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

#define FC_FAST_STEP_MIN 2.0e-5f
#define FC_F16_MAX 65504.0f
#define FC_FMA_BIAS_MAX 512.0f

/* ---------- 共享片段 ---------- */
static inline float16x8_t fc_step_store(const float16x8_t mn[4], const float16x8_t mx[4],
                                        fc_block *out, int g, float32x4_t *lo_mn,
                                        float32x4_t *hi_mn)
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
static void fc_quant_f32(const float16_t *p, fc_block *out, const float32x4_t lo_mn[4],
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
static void fc_fallback(fc_block *out, const float16_t *p, const float16x8_t step16[4],
                        const float32x4_t lo_mn[4], const float32x4_t hi_mn[4])
{
    const float32x4_t zero = vdupq_n_f32(0.0f), one = vdupq_n_f32(1.0f);
    float32x4_t lo_inv[4], hi_inv[4];
    for (int g = 0; g < 4; g++) {
        float32x4_t sl = vcvt_f32_f16(vget_low_f16(step16[g]));
        float32x4_t sh = vcvt_high_f32_f16(step16[g]);
        lo_inv[g] = vbslq_f32(vceqq_f32(sl, zero), zero, vdivq_f32(one, sl));
        hi_inv[g] = vbslq_f32(vceqq_f32(sh, zero), zero, vdivq_f32(one, sh));
    }
    fc_quant_f32(p, out, lo_mn, hi_mn, lo_inv, hi_inv);
}
static inline void fc_inv16(const float16x8_t step16[4], float16x8_t inv16[4])
{
    const float16x8_t zero16 = vdupq_n_f16(0.0f);
    for (int g = 0; g < 4; g++) {
        float16x8_t e = vrecpeq_f16(step16[g]);
        e = vmulq_f16(vrecpsq_f16(step16[g], e), e);
        inv16[g] = vbslq_f16(vceqq_f16(step16[g], zero16), zero16, e);
    }
}
static inline int fc_guard2(const float16x8_t mn[4], const float16x8_t mx[4],
                            const float16x8_t step16[4])
{
    const float16x8_t big = vdupq_n_f16(FC_F16_MAX), zero16 = vdupq_n_f16(0.0f);
    float16x8_t acc = big, rng = zero16;
    for (int g = 0; g < 4; g++) {
        acc = vminq_f16(acc, vbslq_f16(vceqq_f16(step16[g], zero16), big, step16[g]));
        rng = vmaxq_f16(rng, vsubq_f16(mx[g], mn[g]));
    }
    return (float)vminvq_f16(acc) < FC_FAST_STEP_MIN || (float)vmaxvq_f16(rng) > FC_F16_MAX;
}
static inline int fc_guard3(const float16x8_t mn[4], const float16x8_t inv16[4],
                            float16x8_t bias[4])
{
    float16x8_t mag = vdupq_n_f16(0.0f);
    for (int g = 0; g < 4; g++) {
        bias[g] = vnegq_f16(vmulq_f16(mn[g], inv16[g]));
        mag = vmaxq_f16(mag, vabsq_f16(bias[g]));
    }
    return (float)vmaxvq_f16(mag) > FC_FMA_BIAS_MAX;
}
/* 4 行展开 + 树形归约 (当前库使用) */
static inline void fc_minmax_u4(const float16_t *p, float16x8_t mn[4], float16x8_t mx[4])
{
    for (int g = 0; g < 4; g++) {
        float16x8_t a = vld1q_f16(p + g * 8), b = vld1q_f16(p + FC_N + g * 8);
        float16x8_t c = vld1q_f16(p + 2 * FC_N + g * 8), d = vld1q_f16(p + 3 * FC_N + g * 8);
        mn[g] = vminq_f16(vminq_f16(a, b), vminq_f16(c, d));
        mx[g] = vmaxq_f16(vmaxq_f16(a, b), vmaxq_f16(c, d));
    }
    for (int r = 4; r < FC_N; r += 4)
        for (int g = 0; g < 4; g++) {
            float16x8_t a = vld1q_f16(p + r * FC_N + g * 8), b = vld1q_f16(p + (r + 1) * FC_N + g * 8);
            float16x8_t c = vld1q_f16(p + (r + 2) * FC_N + g * 8), d = vld1q_f16(p + (r + 3) * FC_N + g * 8);
            mn[g] = vminq_f16(mn[g], vminq_f16(vminq_f16(a, b), vminq_f16(c, d)));
            mx[g] = vmaxq_f16(mx[g], vmaxq_f16(vmaxq_f16(a, b), vmaxq_f16(c, d)));
        }
}

/* ===== OLD: 优化前 ===== */
static inline void minmax_old(const float16_t *p, float16x8_t mn[4], float16x8_t mx[4])
{
    mn[0] = vld1q_f16(p + 0); mn[1] = vld1q_f16(p + 8);
    mn[2] = vld1q_f16(p + 16); mn[3] = vld1q_f16(p + 24);
    for (int g = 0; g < 4; g++) mx[g] = mn[g];
    for (int r = 1; r < FC_N; r++)
        for (int g = 0; g < 4; g++) {
            float16x8_t v = vld1q_f16(p + r * FC_N + g * 8);
            mn[g] = vminq_f16(mn[g], v);
            mx[g] = vmaxq_f16(mx[g], v);
        }
}
static void comp_OLD(const fc_f16 *restrict in, fc_block *restrict out)
{
    const float16_t *p = (const float16_t *)in;
    float16x8_t mn[4], mx[4], step16[4], inv16[4];
    float32x4_t lo_mn[4], hi_mn[4];
    minmax_old(p, mn, mx);
    for (int g = 0; g < 4; g++) step16[g] = fc_step_store(mn, mx, out, g, &lo_mn[g], &hi_mn[g]);
    if (fc_guard2(mn, mx, step16)) { fc_fallback(out, p, step16, lo_mn, hi_mn); return; }
    fc_inv16(step16, inv16);
    for (int r = 0; r < FC_N; r++) {
        const float16_t *row = p + r * FC_N;
        uint8_t *dst = out->q + r * FC_N;
        for (int g = 0; g < 4; g++) {
            float16x8_t t = vmulq_f16(vsubq_f16(vld1q_f16(row + g * 8), mn[g]), inv16[g]);
            vst1_u8(dst + g * 8, vqmovn_u16(vcvtnq_u16_f16(t)));
        }
    }
}

/* ===== NEW: 当前库 ===== */
static void comp_NEW(const fc_f16 *restrict in, fc_block *restrict out) { fc_compress_neon(in, out); }
static void comp_HP(const fc_f16 *restrict in, fc_block *restrict out) { fc_compress_neon_hp(in, out); }

/* ===== U4: 只做 4 行展开 + 32B store, 量化仍 sub/mul ===== */
static void comp_U4(const fc_f16 *restrict in, fc_block *restrict out)
{
    const float16_t *p = (const float16_t *)in;
    float16x8_t mn[4], mx[4], step16[4], inv16[4];
    float32x4_t lo_mn[4], hi_mn[4];
    fc_minmax_u4(p, mn, mx);
    for (int g = 0; g < 4; g++) step16[g] = fc_step_store(mn, mx, out, g, &lo_mn[g], &hi_mn[g]);
    if (fc_guard2(mn, mx, step16)) { fc_fallback(out, p, step16, lo_mn, hi_mn); return; }
    fc_inv16(step16, inv16);
    for (int r = 0; r < FC_N; r++) {
        const float16_t *row = p + r * FC_N;
        uint8x16x2_t o;
        o.val[0] = vqmovn_high_u16(vqmovn_u16(vcvtnq_u16_f16(vmulq_f16(vsubq_f16(vld1q_f16(row + 0), mn[0]), inv16[0]))),
                                   vcvtnq_u16_f16(vmulq_f16(vsubq_f16(vld1q_f16(row + 8), mn[1]), inv16[1])));
        o.val[1] = vqmovn_high_u16(vqmovn_u16(vcvtnq_u16_f16(vmulq_f16(vsubq_f16(vld1q_f16(row + 16), mn[2]), inv16[2]))),
                                   vcvtnq_u16_f16(vmulq_f16(vsubq_f16(vld1q_f16(row + 24), mn[3]), inv16[3])));
        vst1q_u8_x2(out->q + r * FC_N, o);
    }
}

/* ===== FMA: 只把量化换成 FMA (不做行展开, 仍 4 次 8B store) ===== */
static void comp_FMA(const fc_f16 *restrict in, fc_block *restrict out)
{
    const float16_t *p = (const float16_t *)in;
    float16x8_t mn[4], mx[4], step16[4], inv16[4], bias[4];
    float32x4_t lo_mn[4], hi_mn[4];
    minmax_old(p, mn, mx);
    for (int g = 0; g < 4; g++) step16[g] = fc_step_store(mn, mx, out, g, &lo_mn[g], &hi_mn[g]);
    if (fc_guard2(mn, mx, step16)) { fc_fallback(out, p, step16, lo_mn, hi_mn); return; }
    fc_inv16(step16, inv16);
    if (fc_guard3(mn, inv16, bias)) {
        for (int r = 0; r < FC_N; r++) {
            const float16_t *row = p + r * FC_N;
            uint8_t *dst = out->q + r * FC_N;
            for (int g = 0; g < 4; g++) {
                float16x8_t t = vmulq_f16(vsubq_f16(vld1q_f16(row + g * 8), mn[g]), inv16[g]);
                vst1_u8(dst + g * 8, vqmovn_u16(vcvtnq_u16_f16(t)));
            }
        }
        return;
    }
    for (int r = 0; r < FC_N; r++) {
        const float16_t *row = p + r * FC_N;
        uint8_t *dst = out->q + r * FC_N;
        for (int g = 0; g < 4; g++) {
            float16x8_t t = vfmaq_f16(bias[g], vld1q_f16(row + g * 8), inv16[g]);
            vst1_u8(dst + g * 8, vqmovn_u16(vcvtnq_u16_f16(t)));
        }
    }
}

/* ===== 诊断: 只做 pass1 / 只做 pass2 ===== */
static void comp_P1(const fc_f16 *restrict in, fc_block *restrict out)
{
    const float16_t *p = (const float16_t *)in;
    float16x8_t mn[4], mx[4], step16[4];
    float32x4_t lo_mn[4], hi_mn[4];
    fc_minmax_u4(p, mn, mx);
    for (int g = 0; g < 4; g++) step16[g] = fc_step_store(mn, mx, out, g, &lo_mn[g], &hi_mn[g]);
    if (fc_guard2(mn, mx, step16)) out->q[0] = 1;
}
static void comp_P2(const fc_f16 *restrict in, fc_block *restrict out)
{
    const float16_t *p = (const float16_t *)in;
    float16x8_t mn[4], inv16[4];
    float16x8_t m0 = vdupq_n_f16((float)in[0] * 0.5f);
    float16x8_t i0 = vdupq_n_f16(1.0f / (1.0f + fabsf((float)in[1])));
    for (int g = 0; g < 4; g++) { mn[g] = m0; inv16[g] = i0; }
    for (int r = 0; r < FC_N; r++) {
        const float16_t *row = p + r * FC_N;
        uint8x16x2_t o;
        o.val[0] = vqmovn_high_u16(vqmovn_u16(vcvtnq_u16_f16(vmulq_f16(vsubq_f16(vld1q_f16(row + 0), mn[0]), inv16[0]))),
                                   vcvtnq_u16_f16(vmulq_f16(vsubq_f16(vld1q_f16(row + 8), mn[1]), inv16[1])));
        o.val[1] = vqmovn_high_u16(vqmovn_u16(vcvtnq_u16_f16(vmulq_f16(vsubq_f16(vld1q_f16(row + 16), mn[2]), inv16[2]))),
                                   vcvtnq_u16_f16(vmulq_f16(vsubq_f16(vld1q_f16(row + 24), mn[3]), inv16[3])));
        vst1q_u8_x2(out->q + r * FC_N, o);
    }
}


/* ===== 诊断: 把浮点算术整个挖掉, 看访存本身值多少 ===== */
static void comp_P1L(const fc_f16 *restrict in, fc_block *restrict out) /* pass1 只 load */
{
    const float16_t *p = (const float16_t *)in;
    uint16x8_t a[4];
    for (int g = 0; g < 4; g++) {
        a[g] = vorrq_u16(vreinterpretq_u16_f16(vld1q_f16(p + g * 8)),
                         vreinterpretq_u16_f16(vld1q_f16(p + FC_N + g * 8)));
    }
    for (int r = 2; r < FC_N; r += 2)
        for (int g = 0; g < 4; g++)
            a[g] = vorrq_u16(a[g], vorrq_u16(vreinterpretq_u16_f16(vld1q_f16(p + r * FC_N + g * 8)),
                                             vreinterpretq_u16_f16(vld1q_f16(p + (r + 1) * FC_N + g * 8))));
    for (int g = 0; g < 4; g++) vst1q_u16((uint16_t *)out->min + g * 8, a[g]);
}
static void comp_P2L(const fc_f16 *restrict in, fc_block *restrict out) /* pass2 无浮点算术 */
{
    const float16_t *p = (const float16_t *)in;
    for (int r = 0; r < FC_N; r++) {
        const float16_t *row = p + r * FC_N;
        uint8x16x2_t o;
        o.val[0] = vqmovn_high_u16(vqmovn_u16(vreinterpretq_u16_f16(vld1q_f16(row + 0))),
                                   vreinterpretq_u16_f16(vld1q_f16(row + 8)));
        o.val[1] = vqmovn_high_u16(vqmovn_u16(vreinterpretq_u16_f16(vld1q_f16(row + 16))),
                                   vreinterpretq_u16_f16(vld1q_f16(row + 24)));
        vst1q_u8_x2(out->q + r * FC_N, o);
    }
}

int main(void)
{
    struct { const char *name; comp_fn fn; } V[] = {
        {"OLD", comp_OLD}, {"NEW", comp_NEW}, {"U4", comp_U4}, {"FMA", comp_FMA},
        {"HP", comp_HP}, {"P1", comp_P1}, {"P2", comp_P2}, {"P1L", comp_P1L}, {"P2L", comp_P2L},
    };
    int nv = (int)(sizeof V / sizeof V[0]);

    /* 正确性: 所有变体 vs 标量参考 (少量块, 只看 max|Δq| 与 min/step) */
    printf("=== 正确性 (vs 标量参考, 每 mode 100 随机块) ===\n");
    static fc_f16 in[FC_ELEMS];
    static fc_block br, bt;
    for (int mode = 0; mode < 2; mode++) {
        for (int i = 0; i < nv; i++) {
            if (V[i].name[0] == 'P' || V[i].name[0] == 'H') continue;
            int wq = 0, sd = 0, md = 0;
            for (int k = 0; k < 100; k++) {
                gen(in, 0x9e3779b9 + (uint64_t)k * 7919, mode);
                fc_compress_ref(in, &br);
                V[i].fn(in, &bt);
                for (int j = 0; j < FC_ELEMS; j++) {
                    int d = (int)br.q[j] - (int)bt.q[j];
                    if (d < 0) d = -d;
                    if (d > wq) wq = d;
                }
                if (memcmp(br.step, bt.step, sizeof br.step)) sd = 1;
                if (memcmp(br.min, bt.min, sizeof br.min)) md = 1;
            }
            printf("  mode%d %-3s max|Δq|=%-3d step %s min %s\n", mode, V[i].name, wq,
                   sd ? "DIFF" : "同", md ? "DIFF" : "同");
        }
    }

    int sizes[] = {16, 512};
    for (int si = 0; si < 2; si++) {
        for (int mode = 0; mode < 2; mode++) {
            int nres = sizes[si];
            fc_f16 *arr = aligned_alloc(64, (size_t)nres * 2048);
            fc_block *cb = aligned_alloc(64, (size_t)nres * sizeof(fc_block));
            for (int i = 0; i < nres; i++) gen(arr + (size_t)i * FC_ELEMS, 0x55 + (uint64_t)i, mode);
            for (int i = 0; i < nres; i++) comp_NEW(arr + (size_t)i * FC_ELEMS, &cb[i]);
            int reps = 200000 / nres + 100;
            printf("\n=== 工作集 %d 块, 数据 mode %d (%s) ===\n", nres, mode,
                   mode ? "大 DC, |min| 达 8*range" : "居中, |min| <= 0.5*range");
            for (int i = 0; i < nv; i++) {
                double best = 1e9;
                for (int t = 0; t < 4; t++) {
                    double t0 = now_s();
                    for (int k = 0; k < reps; k++)
                        for (int j = 0; j < nres; j++) V[i].fn(arr + (size_t)j * FC_ELEMS, &cb[j]);
                    double dt = (now_s() - t0) / ((double)reps * nres) * 1e9;
                    if (dt < best) best = dt;
                }
                printf("   %-3s %6.2f ns/块\n", V[i].name, best);
            }
            printf("   (sink=%u)\n", cb[0].q[0]);
            free(arr);
            free(cb);
        }
    }
    return 0;
}
