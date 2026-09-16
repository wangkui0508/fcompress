/* exp_ld.c -- 访存宽度微基准 (解释"为什么压缩是访存受限, 而不是浮点受限")
 *
 *  读: 每块 32 行 x 64B = 2048B, 分别用 16B / 32B / 64B 的 load 指令
 *  写: 每块 32 行 x 32B = 1024B, 分别用 8B / 16B / 32B 的 store 指令
 *  工作集: nres 块 (小 => L1 常驻, 大 => L2/DRAM), 用来看缓存层级的影响
 */
#include "fcompress.h"

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ---- 读: 只做 load + 整数 or 累加 (没有浮点算术) ---- */
static void k_ld16(const float16_t *p, uint16_t *sink)
{
    uint16x8_t a0 = vdupq_n_u16(0), a1 = a0, a2 = a0, a3 = a0;
    for (int r = 0; r < 32; r++) {
        const float16_t *row = p + r * 32;
        a0 = vorrq_u16(a0, vreinterpretq_u16_f16(vld1q_f16(row + 0)));
        a1 = vorrq_u16(a1, vreinterpretq_u16_f16(vld1q_f16(row + 8)));
        a2 = vorrq_u16(a2, vreinterpretq_u16_f16(vld1q_f16(row + 16)));
        a3 = vorrq_u16(a3, vreinterpretq_u16_f16(vld1q_f16(row + 24)));
    }
    vst1q_u16(sink, vorrq_u16(vorrq_u16(a0, a1), vorrq_u16(a2, a3)));
}
static void k_ld32(const float16_t *p, uint16_t *sink)
{
    uint16x8_t a0 = vdupq_n_u16(0), a1 = a0, a2 = a0, a3 = a0;
    for (int r = 0; r < 32; r++) {
        const float16_t *row = p + r * 32;
        float16x8x2_t x = vld1q_f16_x2(row), y = vld1q_f16_x2(row + 16);
        a0 = vorrq_u16(a0, vreinterpretq_u16_f16(x.val[0]));
        a1 = vorrq_u16(a1, vreinterpretq_u16_f16(x.val[1]));
        a2 = vorrq_u16(a2, vreinterpretq_u16_f16(y.val[0]));
        a3 = vorrq_u16(a3, vreinterpretq_u16_f16(y.val[1]));
    }
    vst1q_u16(sink, vorrq_u16(vorrq_u16(a0, a1), vorrq_u16(a2, a3)));
}
static void k_ld64(const float16_t *p, uint16_t *sink)
{
    uint16x8_t a0 = vdupq_n_u16(0), a1 = a0, a2 = a0, a3 = a0;
    for (int r = 0; r < 32; r++) {
        float16x8x4_t v = vld1q_f16_x4(p + r * 32);
        a0 = vorrq_u16(a0, vreinterpretq_u16_f16(v.val[0]));
        a1 = vorrq_u16(a1, vreinterpretq_u16_f16(v.val[1]));
        a2 = vorrq_u16(a2, vreinterpretq_u16_f16(v.val[2]));
        a3 = vorrq_u16(a3, vreinterpretq_u16_f16(v.val[3]));
    }
    vst1q_u16(sink, vorrq_u16(vorrq_u16(a0, a1), vorrq_u16(a2, a3)));
}

/* ---- 写: load + 窄化 u16->u8 + store ---- */
static void k_st8(const float16_t *p, uint8_t *dst)
{
    for (int r = 0; r < 32; r++) {
        uint16x8_t u0 = vreinterpretq_u16_f16(vld1q_f16(p + r * 32 + 0));
        uint16x8_t u1 = vreinterpretq_u16_f16(vld1q_f16(p + r * 32 + 8));
        uint16x8_t u2 = vreinterpretq_u16_f16(vld1q_f16(p + r * 32 + 16));
        uint16x8_t u3 = vreinterpretq_u16_f16(vld1q_f16(p + r * 32 + 24));
        vst1_u8(dst + r * 32 + 0, vqmovn_u16(u0));
        vst1_u8(dst + r * 32 + 8, vqmovn_u16(u1));
        vst1_u8(dst + r * 32 + 16, vqmovn_u16(u2));
        vst1_u8(dst + r * 32 + 24, vqmovn_u16(u3));
    }
}
static void k_st16(const float16_t *p, uint8_t *dst)
{
    for (int r = 0; r < 32; r++) {
        uint16x8_t u0 = vreinterpretq_u16_f16(vld1q_f16(p + r * 32 + 0));
        uint16x8_t u1 = vreinterpretq_u16_f16(vld1q_f16(p + r * 32 + 8));
        uint16x8_t u2 = vreinterpretq_u16_f16(vld1q_f16(p + r * 32 + 16));
        uint16x8_t u3 = vreinterpretq_u16_f16(vld1q_f16(p + r * 32 + 24));
        vst1q_u8(dst + r * 32 + 0, vqmovn_high_u16(vqmovn_u16(u0), u1));
        vst1q_u8(dst + r * 32 + 16, vqmovn_high_u16(vqmovn_u16(u2), u3));
    }
}
static void k_st32(const float16_t *p, uint8_t *dst)
{
    for (int r = 0; r < 32; r++) {
        uint16x8_t u0 = vreinterpretq_u16_f16(vld1q_f16(p + r * 32 + 0));
        uint16x8_t u1 = vreinterpretq_u16_f16(vld1q_f16(p + r * 32 + 8));
        uint16x8_t u2 = vreinterpretq_u16_f16(vld1q_f16(p + r * 32 + 16));
        uint16x8_t u3 = vreinterpretq_u16_f16(vld1q_f16(p + r * 32 + 24));
        uint8x16x2_t b;
        b.val[0] = vqmovn_high_u16(vqmovn_u16(u0), u1);
        b.val[1] = vqmovn_high_u16(vqmovn_u16(u2), u3);
        vst1q_u8_x2(dst + r * 32, b);
    }
}

int main(void)
{
    typedef void (*ld_fn)(const float16_t *, uint16_t *);
    typedef void (*st_fn)(const float16_t *, uint8_t *);
    struct { const char *n; ld_fn f; } LDF[] = {{"ld 16B x4", k_ld16}, {"ld 32B x2", k_ld32}, {"ld 64B x1", k_ld64}};
    struct { const char *n; st_fn f; } STF[] = {{"st 8B x4", k_st8}, {"st 16B x2", k_st16}, {"st 32B x1", k_st32}};

    int sizes[] = {8, 32, 128, 512, 4096};
    for (int si = 0; si < 5; si++) {
        int nres = sizes[si];
        fc_f16 *arr = aligned_alloc(64, (size_t)nres * 2048);
        fc_block *cb = aligned_alloc(64, (size_t)nres * sizeof(fc_block));
        for (int i = 0; i < nres * FC_ELEMS; i++) arr[i] = (fc_f16)(i & 255);
        uint16_t sink[8] = {0};
        int reps = 200000 / nres + 50;
        printf("工作集 %5d 块 (%7.1f KB)\n", nres, nres * (2048.0 + 1152.0) / 1024.0);
        for (int i = 0; i < 3; i++) {
            double best = 1e9;
            for (int t = 0; t < 3; t++) {
                double t0 = now_s();
                for (int k = 0; k < reps; k++)
                    for (int j = 0; j < nres; j++) LDF[i].f((const float16_t *)(arr + (size_t)j * FC_ELEMS), sink);
                double dt = (now_s() - t0) / ((double)reps * nres) * 1e9;
                if (dt < best) best = dt;
            }
            printf("   读 %-10s %6.2f ns/块  %6.1f GB/s\n", LDF[i].n, best, 2048.0 / best);
        }
        for (int i = 0; i < 3; i++) {
            double best = 1e9;
            for (int t = 0; t < 3; t++) {
                double t0 = now_s();
                for (int k = 0; k < reps; k++)
                    for (int j = 0; j < nres; j++) STF[i].f((const float16_t *)(arr + (size_t)j * FC_ELEMS), cb[j].q);
                double dt = (now_s() - t0) / ((double)reps * nres) * 1e9;
                if (dt < best) best = dt;
            }
            printf("   写 %-10s %6.2f ns/块  %6.1f GB/s\n", STF[i].n, best, 1024.0 / best);
        }
        (void)sink[0];
        free(arr);
        free(cb);
    }
    return 0;
}
