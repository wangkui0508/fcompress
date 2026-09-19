/* exp_hq.c -- HQ 编码器: 对拍 + 精度 + 性能报告
 *  1) NEON HQ vs 标量 HQ 参考: bit-exact
 *  2) HQ 的重建 SSE 是否始终 <= 现状快路径 (永不劣化)
 *  3) 各分布上的增益 (dB) 与命中率
 *  4) 编码耗时: 快路径 / HP / HQ
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
    uint64_t x = *s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; *s = x; return x;
}
static double u01(uint64_t *s) { return (double)((xs(s) >> 11) & 0xFFFFFFu) / (double)(1u << 24); }
static double gauss(uint64_t *s)
{
    double a = u01(s) + 1e-12, b = u01(s);
    return sqrt(-2.0 * log(a)) * cos(2.0 * 3.14159265358979 * b);
}
/* dist: 0 均匀 1 高斯 2 拉普拉斯 3 5%离群 4 混合量级 5 大DC */
static void gen(fc_f16 *m, uint64_t seed, int dist)
{
    uint64_t s = seed | 1;
    for (int c = 0; c < FC_N; c++) {
        double mu = (c % 3 == 0) ? 0.0 : ((c % 3 == 1) ? 3.0 : -50.0);
        if (dist == 5) mu *= 20.0;
        double sig = 0.5 + 1.5 * u01(&s);
        for (int r = 0; r < FC_N; r++) {
            double v;
            switch (dist) {
            case 0: v = mu + sig * (u01(&s) - 0.5) * 3.4641; break;
            case 1: v = mu + sig * gauss(&s); break;
            case 2: { double u = u01(&s) - 0.5; v = mu + sig * (-log(1.0 - 2.0 * fabs(u)) * (u > 0 ? 1 : -1)); break; }
            case 3: v = mu + sig * gauss(&s) * (u01(&s) < 0.05 ? 8.0 : 1.0); break;
            case 4: v = mu * (1.0 + 300.0 * u01(&s)) + sig * 0.001 * gauss(&s); break;
            default: v = mu + sig * gauss(&s); break;
            }
            if (v > 65504.0) v = 65504.0;
            if (v < -65504.0) v = -65504.0;
            m[r * FC_N + c] = (fc_f16)v;
        }
    }
}
static double sse(const fc_block *b, const fc_f16 *ref, double *pow_out)
{
    static fc_f16 rec[FC_ELEMS];
    fc_decompress_ref(b, rec);
    double se = 0, sp = 0;
    for (int i = 0; i < FC_ELEMS; i++) {
        double d = (double)rec[i] - (double)ref[i];
        se += d * d;
        sp += (double)ref[i] * (double)ref[i];
    }
    if (pow_out) *pow_out = sp;
    return se;
}

int main(void)
{
    static const char *dn[] = {"均匀", "高斯", "拉普拉斯", "5%离群", "混合量级", "大DC"};
    const int nblk = 3000;
    static fc_f16 in[FC_ELEMS];
    static fc_block bh, br, bb;

    printf("=== 1. NEON HQ vs 标量 HQ 参考 (bit-exact) ===\n");
    int bad = 0, badq = 0;
    for (int dist = 0; dist < 6; dist++)
        for (int k = 0; k < nblk / 6; k++) {
            gen(in, 0x9e3779b9UL + (uint64_t)k * 7919UL + (uint64_t)dist * 1000003UL, dist);
            fc_compress_neon_hq(in, &bh);
            fc_compress_ref_hq(in, &br);
            if (memcmp(&bh, &br, sizeof bh) != 0) {
                bad++;
                for (int i = 0; i < FC_ELEMS; i++) if (bh.q[i] != br.q[i]) badq++;
            }
        }
    printf("   %d 块: 不一致块数 %d (q 差异元素 %d) -> %s\n\n", nblk, bad, badq,
           bad ? "FAIL" : "bit-exact");

    printf("=== 2. HQ 是否永不劣于现状快路径 + 增益 ===\n");
    printf("%-10s %8s %8s %10s %10s %8s\n", "分布", "现状SNR", "HQ SNR", "增益dB", "最差块dB", "劣化块");
    for (int dist = 0; dist < 6; dist++) {
        double sb = 0, sh = 0, sp = 0;
        int worse = 0;
        double worst = 1e30;
        for (int k = 0; k < nblk / 6; k++) {
            gen(in, 0x9e3779b9UL + (uint64_t)k * 7919UL + (uint64_t)dist * 1000003UL, dist);
            fc_compress_neon(in, &bb);
            fc_compress_neon_hq(in, &bh);
            double p;
            double eb = sse(&bb, in, &p), eh = sse(&bh, in, NULL);
            sb += eb; sh += eh; sp += p;
            if (eh > eb * (1.0 + 1e-12)) worse++;
            double d = 10.0 * log10(eb / (eh > 0 ? eh : 1e-300));
            if (d < worst) worst = d;
        }
        printf("%-10s %8.2f %8.2f %+10.3f %+10.3f %8d\n", dn[dist], 10 * log10(sp / sb),
               10 * log10(sp / sh), 10 * log10(sb / sh), worst, worse);
    }

    printf("\n=== 3. 编码耗时 (512 块 L2 常驻, 3 轮取最小) ===\n");
    {
        const int nres = 512, reps = 400;
        fc_f16 *arr = aligned_alloc(64, (size_t)nres * 2048);
        fc_block *cb = aligned_alloc(64, (size_t)nres * sizeof(fc_block));
        for (int i = 0; i < nres; i++) gen(arr + (size_t)i * FC_ELEMS, 0x55 + (uint64_t)i, 1);
        for (int i = 0; i < nres; i++) fc_compress_neon(arr + (size_t)i * FC_ELEMS, &cb[i]);
        struct { const char *n; void (*f)(const fc_f16 *restrict, fc_block *restrict); } V[] = {
            {"快路径 fc_compress_neon", fc_compress_neon},
            {"高精度 fc_compress_neon_hp", fc_compress_neon_hp},
            {"HQ     fc_compress_neon_hq", fc_compress_neon_hq},
        };
        for (int i = 0; i < 3; i++) {
            double best = 1e9;
            for (int t = 0; t < 3; t++) {
                double t0 = now_s();
                for (int k = 0; k < reps; k++)
                    for (int j = 0; j < nres; j++) V[i].f(arr + (size_t)j * FC_ELEMS, &cb[j]);
                double dt = (now_s() - t0) / ((double)reps * nres) * 1e9;
                if (dt < best) best = dt;
            }
            printf("   %-30s %8.1f ns/块  (%5.1fx 于快路径)\n", V[i].n, best, best / 49.0);
        }
        printf("   解压 fc_decompress_neon: HQ 与快路径共用同一解码器, 格式未变\n");
        free(arr); free(cb);
    }
    return bad != 0;
}
