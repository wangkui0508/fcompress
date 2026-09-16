/* test_fcompress.c -- 正确性对拍 + 重建精度 + 性能
 *
 * 用法: ./test_fcompress [stream_blocks]   默认 8192 块 (~16MB)
 */
#include "fcompress.h"

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

/* 生成覆盖各种数量级的 32x32 矩阵 */
static void gen(fc_f16 *m, uint64_t seed, int pattern)
{
    uint64_t s = seed | 1;
    static const double mags[6] = {1e-3, 1.0, 1e3, 60.0, 6e-5, 1000.0};

    for (int c = 0; c < FC_N; c++) {
        double base, range;
        switch (pattern) {
        case 0: base = 0.0; range = 1.0; break;
        case 1: base = (double)(c - 16) * 0.5; range = 1.0 + 0.25 * c; break;
        case 2:
            base = ((c & 1) ? -1.0 : 1.0) * mags[c % 6];
            range = mags[(c / 2) % 6] * 0.5;
            break;
        case 3: base = 1.0; range = 0.0; break; /* 常量列 */
        case 4: base = 3.0; range = 1e-2; break;
        case 5: base = 0.0; range = 5e-4; break; /* step 太小 -> 触发快路径回退 */
        /* range > 65504: f16 域 (v-min) 会溢出成 inf, 必须回退, 否则顶部整段
         * 被饱和到 q=255 (实测: 修复前这三个 pattern 的 max|Δq| 分别是 1/113/127,
         *  ΔSNR -0.34/-42.84/-43.97 dB) */
        case 6: base = -32752.0; range = 65504.0; break;  /* 满量程, 恰好在边界上 */
        case 7: base = -60000.0; range = 120000.0; break; /* range = 120000 */
        case 8: base = -65504.0; range = 131008.0; break; /* f16 全域, range 最大 */
        /* 守卫 3 的边界: |min*inv| 落在 [400, 512] —— 恰好还在 FMA 快路径上,
         * 但 bias 的 f16 舍入已经接近最大 (整列 q 最多平移 0.25 个台阶).
         * 覆盖"FMA 抵消"最坏情况, 40 万块实测 max|Δq| = 1, 最差 ΔSNR -0.57 dB */
        case 9:
            base = ((c & 1) ? -1.0 : 1.0) * (1.5 + 0.7 * (double)(c % 5) / 4.0) * (0.25 + 8.0 * (double)((c / 5) % 4) / 3.0);
            range = 0.25 + 8.0 * (double)((c / 5) % 4) / 3.0;
            break;
        default: base = 3.0; range = 1e-2; break;
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

static int first_diff(const uint8_t *a, const uint8_t *b, int n)
{
    for (int i = 0; i < n; i++)
        if (a[i] != b[i]) return i;
    return -1;
}

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

/* 反量化结果与参考实现的 ULP 差 (f16 位模式是有序的, 可直接当整数比) */
static int max_ulp16(const fc_f16 *a, const fc_f16 *b)
{
    int worst = 0;
    for (int i = 0; i < FC_ELEMS; i++) {
        uint16_t x, y;
        memcpy(&x, &a[i], 2);
        memcpy(&y, &b[i], 2);
        int d = (int)x - (int)y;
        if (d < 0) d = -d;
        if (d > worst) worst = d;
    }
    return worst;
}

int main(int argc, char **argv)
{
    const int nstream = (argc > 1) ? atoi(argv[1]) : 8192;
    int fails = 0;

    printf("=== 1. NEON 高精度路径 (f32 量化) vs 标量参考: bit-exact ===\n");
    for (int pat = 0; pat < 10; pat++) {
        fc_f16 in[FC_ELEMS];
        static fc_block bn, br;
        gen(in, 0x1234567 + pat, pat);

        fc_compress_neon_hp(in, &bn);
        fc_compress_ref(in, &br);

        int dq = first_diff(bn.q, br.q, FC_ELEMS);
        int dmin = memcmp(bn.min, br.min, sizeof(bn.min));
        int dstep = memcmp(bn.step, br.step, sizeof(bn.step));

        fc_f16 on[FC_ELEMS], orr[FC_ELEMS];
        fc_decompress_neon(&bn, on);
        fc_decompress_ref(&bn, orr);
        int ulp = max_ulp16(on, orr);
        double snr = snr_db(in, on);

        printf("  pattern %d: q %s(off %d)  min %s  step %s  | dequant-vs-ref ulp=%d  "
               "| recon SNR=%6.2f dB\n",
               pat, dq < 0 ? "EXACT " : "DIFF!!", dq,
               dmin == 0 ? "EXACT" : "DIFF!",
               dstep == 0 ? "EXACT" : "DIFF!",
               ulp, snr);
        if (dq >= 0 || dmin || dstep || ulp > 2) fails++;
    }

    printf("\n=== 1b. NEON 默认快路径 (f16 域量化) vs 参考 ===\n");
    {
        long tot_dq = 0;
        int max_dq = 0, bad_step = 0;
        double worst_dsnr = 1e30;
        for (int pat = 0; pat < 10; pat++) {
            fc_f16 in[FC_ELEMS];
            fc_block bf, br;
            gen(in, 0x1234567 + pat, pat);
            fc_compress_neon(in, &bf);
            fc_compress_ref(in, &br);

            int nd = 0, md = 0;
            for (int i = 0; i < FC_ELEMS; i++) {
                int d = (int)bf.q[i] - (int)br.q[i];
                if (d < 0) d = -d;
                if (d) nd++;
                if (d > md) md = d;
                tot_dq += d;
            }
            for (int c = 0; c < FC_N; c++)
                if (memcmp(&bf.step[c], &br.step[c], 2) != 0) bad_step++;

            fc_f16 of[FC_ELEMS], orf[FC_ELEMS];
            fc_decompress_neon(&bf, of);
            fc_decompress_neon(&br, orf);
            double d = snr_db(in, of) - snr_db(in, orf);
            if (d < worst_dsnr) worst_dsnr = d;
            if (md > max_dq) max_dq = md;
            printf("  pattern %d: q 差异 %4d/1024 (max %d)  step %s  ΔSNR %+6.2f dB\n", pat, nd, md,
                   bad_step ? "DIFF!" : "与参考逐位相同", d);
        }
        printf("  -> Σ|Δq|=%ld, max|Δq|=%d, step %s, 最差 ΔSNR %+.2f dB\n", tot_dq, max_dq,
               bad_step == 0 ? "全部逐位相同" : "有差异!", worst_dsnr);
        if (max_dq > 1 || bad_step) fails++;
    }

    printf("\n=== 2. 列级重建边界检查 (pattern 1) ===\n");
    {
        fc_f16 in[FC_ELEMS], out[FC_ELEMS];
        fc_block b;
        gen(in, 0xABCDEF, 1);
        fc_compress_neon(in, &b);
        fc_decompress_neon(&b, out);
        double worst_step_frac = 0.0;
        int bad = 0;
        for (int c = 0; c < FC_N; c++) {
            float mn = 1e30f, mx = -1e30f, rmn = 1e30f, rmx = -1e30f;
            for (int r = 0; r < FC_N; r++) {
                float v = (float)in[r * FC_N + c], w = (float)out[r * FC_N + c];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                if (w < rmn) rmn = w;
                if (w > rmx) rmx = w;
            }
            double err = fabs((double)rmn - (double)mn);
            err = fmax(err, fabs((double)rmx - (double)mx));
            double st = (double)b.step[c];
            double frac = (st > 0.0) ? err / st : 0.0; /* 理论上界 0.5 + f16 域舍入余量 */
            if (frac > worst_step_frac) worst_step_frac = frac;
            if (frac > 0.6) bad++;
            if ((float)b.min[c] != mn) bad++;
        }
        printf("  min 精确复现: %s\n", bad == 0 ? "yes" : "NO");
        printf("  端点最大偏差 = %.4f * step (理论上界 0.5, f16 量化再叠加 ~0.06) -> %s\n",
               worst_step_frac, worst_step_frac <= 0.6 ? "OK" : "超界!!");
        if (bad) fails++;
    }

    printf("\n=== 2b. FP16 step 的精度边界 (pattern 2, 混数量级) ===\n");
    {
        fc_f16 in[FC_ELEMS];
        fc_block b;
        gen(in, 0x1234567 + 2, 2);
        fc_compress_neon(in, &b);

        printf("   列   range(f32)     step=range/255   f16(step)       相对误差   f16 状态\n");
        int nsub = 0;
        double worst_rel = 0.0;
        for (int c = 0; c < FC_N; c++) {
            float mn = 1e30f, mx = -1e30f;
            for (int r = 0; r < FC_N; r++) {
                float v = (float)in[r * FC_N + c];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            double range = (double)mx - (double)mn;
            double s32 = range / 255.0;
            double s16 = (double)b.step[c];
            double rel = (s32 > 0.0) ? fabs(s16 - s32) / s32 : 0.0;
            int sub = (s16 != 0.0 && s16 < 6.1035e-5); /* f16 次正规阈值 2^-14 */
            if (sub) nsub++;
            if (rel > worst_rel) worst_rel = rel;
            if (c % 4 == 0 || sub)
                printf("  %3d  %10.3e   %10.3e   %10.3e   %8.2e   %s\n", c, range, s32, s16, rel,
                       s16 == 0.0 ? "=0 (常量列)" : (sub ? "次正规! 精度骤降" : "正常"));
        }
        printf("  -> 次正规/零 step 列数: %d/32,  最大相对误差 %.2e\n", nsub, worst_rel);
        printf("     f16 次正规只有 1~2 位有效位; step 小到 2^-14 以下时重建误差会明显变大.\n");
    }

    printf("\n=== 3. 性能 ===\n");
    {
        volatile uint64_t sink = 0;
        const int nres = 512; /* 512*2KB = 1MB 输入, 576KB 输出 -> L2 常驻 */
        const int reps = 4000;

        fc_f16 *arr = aligned_alloc(64, (size_t)nres * 2048);
        fc_block *cb = aligned_alloc(64, (size_t)nres * sizeof(fc_block));
        fc_f16 *ob = aligned_alloc(64, (size_t)nres * 2048);
        for (int i = 0; i < nres; i++) gen(arr + (size_t)i * FC_ELEMS, 0x55 + i, 1);

        /* 预热 */
        for (int i = 0; i < nres; i++) {
            fc_compress_neon(arr + (size_t)i * FC_ELEMS, &cb[i]);
            fc_decompress_neon(&cb[i], ob + (size_t)i * FC_ELEMS);
        }

        /* 两段计时完全分开, 且结果被 checksum 消费, 编译器无法提升/融合.
         * 各跑 3 轮取最小: 单次计时受频率爬升/调度影响有 ±5% 抖动,
         * 取最小后与 exp_opt.c 的多轮最小值可比. */
        double tc = 1e9, td = 1e9;
        for (int round = 0; round < 3; round++) {
            double t0 = now_s();
            for (int k = 0; k < reps; k++)
                for (int i = 0; i < nres; i++)
                    fc_compress_neon(arr + (size_t)i * FC_ELEMS, &cb[i]);
            double t1 = now_s();
            for (int k = 0; k < reps; k++)
                for (int i = 0; i < nres; i++)
                    fc_decompress_neon(&cb[i], ob + (size_t)i * FC_ELEMS);
            double t2 = now_s();
            if (t1 - t0 < tc) tc = t1 - t0;
            if (t2 - t1 < td) td = t2 - t1;
        }
        double t0 = 0.0, t1 = tc, t2 = tc + td;

        uint64_t h = 0;
        for (int i = 0; i < nres; i++)
            h += cb[i].q[(i * 7) & 1023] +
                 (uint64_t)ob[(size_t)i * FC_ELEMS + ((i * 13) & 1023)];
        sink = h;

        const long n = (long)reps * nres;
        const double bytes = 2048.0 + 1152.0; /* 每块读+写 */
        printf("  L2 常驻 %d 块 x %d 轮:\n", nres, reps);
        printf("    compress  : %8.2f ns/block  %6.2f GB/s  %5.3f ns/元素\n",
               (t1 - t0) / n * 1e9, bytes * n / (t1 - t0) / 1e9, (t1 - t0) / n * 1e9 / 1024.0);
        printf("    decompress: %8.2f ns/block  %6.2f GB/s  %5.3f ns/元素\n",
               (t2 - t1) / n * 1e9, bytes * n / (t2 - t1) / 1e9, (t2 - t1) / n * 1e9 / 1024.0);

        free(arr);
        free(cb);
        free(ob);

        /* 流式: 远超 L2, 看 DRAM 带宽下每块成本 */
        fc_f16 *big = aligned_alloc(64, (size_t)nstream * 2048);
        fc_block *bb = aligned_alloc(64, (size_t)nstream * sizeof(fc_block));
        fc_f16 *bo = aligned_alloc(64, (size_t)nstream * 2048);
        for (int i = 0; i < nstream; i++) gen(big + (size_t)i * FC_ELEMS, 1234 + i, 1);

        double t3 = now_s();
        for (int i = 0; i < nstream; i++) fc_compress_neon(big + (size_t)i * FC_ELEMS, &bb[i]);
        double t4 = now_s();
        for (int i = 0; i < nstream; i++) fc_decompress_neon(&bb[i], bo + (size_t)i * FC_ELEMS);
        double t5 = now_s();

        uint64_t h2 = 0;
        for (int i = 0; i < nstream; i += 64)
            h2 += bb[i].q[i & 1023] + (uint64_t)bo[(size_t)i * FC_ELEMS + (i & 1023)];
        sink = h2;

        double mb = (double)nstream * 2048.0;
        printf("  流式 %d 块 (%.1f MB, 超出 L2):\n", nstream, mb / 1048576.0);
        printf("    compress  : %8.2f ns/block  %6.2f GB/s\n",
               (t4 - t3) / nstream * 1e9, bytes * nstream / (t4 - t3) / 1e9);
        printf("    decompress: %8.2f ns/block  %6.2f GB/s\n",
               (t5 - t4) / nstream * 1e9, bytes * nstream / (t5 - t4) / 1e9);
        free(big);
        free(bb);
        free(bo);
        printf("  (sink=%llu)\n", (unsigned long long)sink);
    }

    printf("\n=== 4. 压缩比 ===\n");
    printf("  原始   : 2048 B (32x32x2B)\n");
    printf("  压缩后 : 1024 B(q) + 64 B(min) + 64 B(step) = 1152 B\n");
    printf("  压缩比 : %.3fx  (规模无关: 每 32 个值固定 32B+4B 开销)\n",
           2048.0 / 1152.0);

    printf("\n%s\n", fails == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return fails != 0;
}
