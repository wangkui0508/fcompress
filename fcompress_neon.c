/* fcompress_neon.c -- NEON 实现 (Apple Silicon, FP16 向量算术)
 *
 * 两条编码路径, 共用同一套 min/step 计算:
 *
 *   fc_compress_neon     默认快路径. 量化在 f16 域完成:
 *                        (v-min)*inv 用 FSUB.8H/FMUL.8H, 再一条 FCVTNU.8H
 *                        直接得到 u16 (就近偶数+饱和), 每 8 元素 6 条指令.
 *                        比 f32 路径快约 1.9x, q 最多差 1 个 LSB.
 *                        两种情况下自动回退 f32 路径:
 *                          a) 某列 step 小到 1/step 会溢出 f16;
 *                          b) 某列跨度 mx-min 超过 f16 最大值 (否则 f16 域里
 *                             v-min 会溢出成 inf, 顶部整段被饱和到 255).
 *
 *   fc_compress_neon_hp  高精度路径. 量化在 f32 域 (FCVTNU.4S + 饱和窄化),
 *                        与 fcompress_ref.c 逐位一致.
 *
 * 两条路径的 step/min 完全相同 (同一段代码算的), 所以落盘数据只在 q 上有
 * ±1 的差别.
 *
 * 指令要点:
 *   1) min/max 规约在 f16 域 (vminq_f16/vmaxq_f16), 精确无舍入; 每行 32 个
 *      f16 = 4 个 float16x8_t, 一趟扫描 + 4 组累加器即得全部 32 列的 min/max,
 *      不需要转置, 不需要水平规约. (拆成 8 组累加器打断链式依赖在本机实测
 *      无收益: vmin 延迟短, 4 组交错已经喂饱流水线, 反而多出寄存器压力 -
 *      完整 A/B 见 exp_ab.c)
 *   2) step = (max-min)/255 用 vdivq_f32 (IEEE 精确) 再窄化存 f16.
 *      vdiv 在这里是免费的: 8 条独立的 VDIV.4S 延迟被后面 pass 2 的计算完全掩盖.
 *      实测换成 vrecpeq_f32+Newton 反而慢 1%, 换 vrecpeq_f16 则无法满足精度.
 *   3) 编码用的 inv 由"落盘的那个 step16"取倒数, 保证编解码网格一致.
 *      step==0 (常量列) 时 inv=0, 避免 inf/NaN.
 *   4) 饱和链: FP->uint 转换对负数饱和到 0, vqmovn_u32/vqmovn_u16 再夹到 255,
 *      不需要任何显式 clamp. (ARMv8 无直接 f16->u32 转换的 ACLE intrinsic,
 *      但 f16->u16 有, 这正是快路径能成立的原因.)
 */
#include "fcompress.h"

#include <arm_neon.h>

#if !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error "需要 FP16 向量算术: 用 -mcpu=apple-m1 及以上 (或 -march=armv8.6-a+fp16) 编译"
#endif

/* 快路径守卫 1: inv = 1/step 必须能装进 f16 (max 65504), 留 ~24% 余量.
 * step < 2e-5  <=>  range < 5.1e-3. 这么小的 range 本来已经贴到本格式的
 * 精度下限了, 直接走 f32 路径更稳. step == 0 不触发 (两条路径都正确). */
#define FC_FAST_STEP_MIN 2.0e-5f

/* 快路径守卫 2: 量化里的 (v - min) 在 f16 域完成, 而 f16 最大只到 65504.
 * 某列跨度超过 f16 最大有限值时 v-min 溢出成 +inf, inf*inv 仍是 inf,
 * FCVTNU 把它饱和到 65535 -> vqmovn 夹到 255, 于是该列顶部整段被压成 255.
 * 实测 (min=-60000, max=60000, range=120000): max|dq| = 115,
 * 重建 SNR 从 48.5 dB 崩到 4.3 dB.
 * 判据直接看 f16 域的 mx-mn 是否溢出 —— 这正是量化时实际算的那个差值,
 * 比用 f32 判 range > 65504 更精确: range 落在 (65504, 65520) 时 f16 差仍然
 * 有限, 只是损失 < 1 个 LSB, 不必回退. */
#define FC_F16_MAX 65504.0f

/* ---------- 一趟扫描求 32 列 min/max (f16 域, 精确) ---------- */
static inline void fc_minmax(const float16_t *p, float16x8_t mn[4], float16x8_t mx[4])
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

/* ---------- 第 g 组 8 列: 算 step, 落盘 min/step, 顺便吐出 f32 的 min ---------- */
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

/* ---------- f32 域量化: 每 8 元素 ~13 条指令 ---------- */
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

/* ---------- f16 域量化: 每 8 元素 6 条指令 ---------- */
static void fc_quant_f16(const float16_t *p, fc_block *out, const float16x8_t mn[4],
                         const float16x8_t inv[4])
{
    for (int r = 0; r < FC_N; r++) {
        const float16_t *row = p + r * FC_N;
        uint8_t *dst = out->q + r * FC_N;
        for (int g = 0; g < 4; g++) {
            float16x8_t v = vld1q_f16(row + g * 8);
            float16x8_t t = vmulq_f16(vsubq_f16(v, mn[g]), inv[g]);
            vst1_u8(dst + g * 8, vqmovn_u16(vcvtnq_u16_f16(t)));
        }
    }
}

void fc_compress_neon(const fc_f16 *restrict in, fc_block *restrict out)
{
    const float16_t *p = (const float16_t *)in;
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t one = vdupq_n_f32(1.0f);

    float16x8_t mn[4], mx[4], step16[4];
    float32x4_t lo_mn[4], hi_mn[4], lo_inv[4], hi_inv[4];

    fc_minmax(p, mn, mx);
    for (int g = 0; g < 4; g++)
        step16[g] = fc_step_store(mn, mx, out, g, &lo_mn[g], &hi_mn[g]);

    /* 守卫 1: 把 step==0 换成 65504 后取全部 32 列的最小值, 看 1/step 会不会溢出 f16.
     * 守卫 2: 取全部 32 列 f16 域跨度 mx-mn 的最大值, 看它会不会溢出成 inf. */
    const float16x8_t big = vdupq_n_f16(FC_F16_MAX);
    const float16x8_t zero16 = vdupq_n_f16(0.0f);
    float16x8_t acc = big, rng = zero16;
    for (int g = 0; g < 4; g++) {
        acc = vminq_f16(acc, vbslq_f16(vceqq_f16(step16[g], zero16), big, step16[g]));
        /* inf 在 vmax 里会可靠地胜出, 所以不需要额外的 isinf 判断.
         * (NaN 输入不支持, 见 fcompress.h, 不在守卫覆盖范围内) */
        rng = vmaxq_f16(rng, vsubq_f16(mx[g], mn[g]));
    }

    if ((float)vminvq_f16(acc) < FC_FAST_STEP_MIN || (float)vmaxvq_f16(rng) > FC_F16_MAX) {
        /* min/step 已在上面落盘, 这里只把量化域换回 f32 */
        for (int g = 0; g < 4; g++) {
            float32x4_t sl = vcvt_f32_f16(vget_low_f16(step16[g]));
            float32x4_t sh = vcvt_high_f32_f16(step16[g]);
            lo_inv[g] = vbslq_f32(vceqq_f32(sl, zero), zero, vdivq_f32(one, sl));
            hi_inv[g] = vbslq_f32(vceqq_f32(sh, zero), zero, vdivq_f32(one, sh));
        }
        fc_quant_f32(p, out, lo_mn, hi_mn, lo_inv, hi_inv);
        return;
    }

    /* f16 倒数 + 1 次 Newton-Raphson; 约 2^-11 相对精度, 对 255 级量化远远够用 */
    float16x8_t inv16[4];
    for (int g = 0; g < 4; g++) {
        float16x8_t e = vrecpeq_f16(step16[g]);
        e = vmulq_f16(vrecpsq_f16(step16[g], e), e);
        inv16[g] = vbslq_f16(vceqq_f16(step16[g], zero16), zero16, e);
    }
    fc_quant_f16(p, out, mn, inv16);
}

void fc_compress_neon_hp(const fc_f16 *restrict in, fc_block *restrict out)
{
    const float16_t *p = (const float16_t *)in;
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t one = vdupq_n_f32(1.0f);

    float16x8_t mn[4], mx[4], step16[4];
    float32x4_t lo_mn[4], hi_mn[4], lo_inv[4], hi_inv[4];

    fc_minmax(p, mn, mx);
    for (int g = 0; g < 4; g++) {
        step16[g] = fc_step_store(mn, mx, out, g, &lo_mn[g], &hi_mn[g]);
        float32x4_t sl = vcvt_f32_f16(vget_low_f16(step16[g]));
        float32x4_t sh = vcvt_high_f32_f16(step16[g]);
        lo_inv[g] = vbslq_f32(vceqq_f32(sl, zero), zero, vdivq_f32(one, sl));
        hi_inv[g] = vbslq_f32(vceqq_f32(sh, zero), zero, vdivq_f32(one, sh));
    }
    fc_quant_f32(p, out, lo_mn, hi_mn, lo_inv, hi_inv);
}

void fc_decompress_neon(const fc_block *restrict in, fc_f16 *restrict out)
{
    float16_t *dst = (float16_t *)out;

    for (int r = 0; r < FC_N; r++) {
        const uint8_t *qrow = in->q + r * FC_N;
        float16_t *orow = dst + r * FC_N;
        for (int g = 0; g < 4; g++) {
            /* min/step 的 8 条向量 load 会被提出行循环, 内层只剩 5 条 */
            uint8x8_t x8 = vld1_u8(qrow + g * 8);
            uint16x8_t x16 = vmovl_u8(x8);
            float16x8_t xf = vcvtq_f16_u16(x16); /* u16 -> f16, 精确 */
            float16x8_t mnv = vld1q_f16((const float16_t *)in->min + g * 8);
            float16x8_t stv = vld1q_f16((const float16_t *)in->step + g * 8);
            vst1q_f16(orow + g * 8, vfmaq_f16(mnv, xf, stv)); /* min + q*step, 单次舍入 */
        }
    }
}
