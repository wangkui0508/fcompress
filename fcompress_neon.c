/* fcompress_neon.c -- NEON 实现 (Apple Silicon, FP16 向量算术)
 *
 * 两条编码路径, 共用同一套 min/step 计算:
 *
 *   fc_compress_neon     默认快路径. 量化在 f16 域完成, 有两种形式:
 *                          FMA 型 (默认): t = v*inv + bias, bias = -min*inv
 *                                        预计算, 每 8 元素 5 条指令;
 *                          sub+mul 型:   t = (v-min)*inv, 每 8 元素 6 条指令
 *                                        (|min| 相对 range 太大时的回退).
 *                        再一条 FCVTNU.8H 得到 u16 (就近偶数+饱和).
 *                        比 f32 路径快约 2.2x, q 最多差 1 个 LSB.
 *                        三种情况下自动回退 (a/b 回退到 f32, c 回退到 sub+mul):
 *                          a) 某列 step 小到 1/step 会溢出 f16;
 *                          b) 某列跨度 mx-min 超过 f16 最大值 (否则 f16 域里
 *                             v-min 会溢出成 inf, 顶部整段被饱和到 255);
 *                          c) 某列 |min| > 2*range, FMA 会抵消掉有效位.
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
 *      不需要转置, 不需要水平规约. 扫描按 4 行展开 + 组内树形归约, 把依赖链
 *      从 31 级压到 8 级 (实测 -5%, 完整 A/B 见 exp_opt.c).
 *      (拆成 8 组累加器在本机实测无收益: vmin 延迟短, 4 组交错已经喂饱流水线,
 *      反而多出寄存器压力 - 完整 A/B 见 exp_ab.c)
 *   2) step = (max-min)/255 用 vdivq_f32 (IEEE 精确) 再窄化存 f16.
 *      vdiv 在这里是免费的: 8 条独立的 VDIV.4S 延迟被后面 pass 2 的计算完全掩盖.
 *      实测换成 vrecpeq_f32+Newton 反而慢 1%, 换 vrecpeq_f16 则无法满足精度.
 *   3) 编码用的 inv 由"落盘的那个 step16"取倒数, 保证编解码网格一致.
 *      step==0 (常量列) 时 inv=0, 避免 inf/NaN.
 *   4) 饱和链: FP->uint 转换对负数饱和到 0, vqmovn_u32/vqmovn_u16 再夹到 255,
 *      不需要任何显式 clamp. (ARMv8 无直接 f16->u32 转换的 ACLE intrinsic,
 *      但 f16->u16 有, 这正是快路径能成立的原因.)
 *   5) 这个 kernel 是**访存受限**的, 不是浮点受限: 只读 pass 1 (2048B) 就要
 *      ~20 ns, pass 2 (再读 2048B + 窄化 + 写 1024B) 要 ~32 ns, 而把 pass 1
 *      的 256 条 FMIN/FMAX 全部删掉、或把 pass 2 的 FCVTNU 全部删掉, 耗时都
 *      几乎不变 (见 exp_opt.c 的 P1/P2 与 exp_ld.c). 所以优化方向是:
 *      少几条指令 + 把访存等得更快, 而不是"少几次浮点运算".
 *      加载宽度不是越大越好: 实测 4x16B 比 1x64B (vld1q_f16_x4) 快 ~15%
 *      (512 块工作集下 16.0 vs 18.6 ns), 所以保持 16B 加载.
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

/* 快路径守卫 3: FMA 量化用的 bias = -mn*inv 必须能有意义地装进 f16.
 * 阈值 512 <=> |mn| <= 2.01*range (见 fc_compress_neon 里的推导).
 * 触发后回退到 f16 的 sub+mul 量化, 只慢约 6%. */
#define FC_FMA_BIAS_MAX 512.0f

/* ---------- 一趟扫描求 32 列 min/max (f16 域, 精确) ----------
 * 4 行展开 + 组内树形归约: 把 mn/mx 的串行依赖链从 31 级压到 8 级, 实测比
 * 单行版本快 ~5% (见 exp_opt.c 的 OLD/U4 对照).
 * 注意: 加载宽度不是越大越好 —— 实测 4x16B 比 1x64B(vld1q_f16_x4) 更快
 * (512 块工作集下 16.0 vs 18.6 ns, 见 exp_ld.c), 所以这里保持 16B 加载. */
static inline void fc_minmax(const float16_t *p, float16x8_t mn[4], float16x8_t mx[4])
{
    for (int g = 0; g < 4; g++) {
        float16x8_t a = vld1q_f16(p + g * 8);
        float16x8_t b = vld1q_f16(p + FC_N + g * 8);
        float16x8_t c = vld1q_f16(p + 2 * FC_N + g * 8);
        float16x8_t d = vld1q_f16(p + 3 * FC_N + g * 8);
        mn[g] = vminq_f16(vminq_f16(a, b), vminq_f16(c, d));
        mx[g] = vmaxq_f16(vmaxq_f16(a, b), vmaxq_f16(c, d));
    }

    for (int r = 4; r < FC_N; r += 4) {
        for (int g = 0; g < 4; g++) {
            float16x8_t a = vld1q_f16(p + r * FC_N + g * 8);
            float16x8_t b = vld1q_f16(p + (r + 1) * FC_N + g * 8);
            float16x8_t c = vld1q_f16(p + (r + 2) * FC_N + g * 8);
            float16x8_t d = vld1q_f16(p + (r + 3) * FC_N + g * 8);
            mn[g] = vminq_f16(mn[g], vminq_f16(vminq_f16(a, b), vminq_f16(c, d)));
            mx[g] = vmaxq_f16(mx[g], vmaxq_f16(vmaxq_f16(a, b), vmaxq_f16(c, d)));
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

/* ---------- f16 域量化, 两种形式 ----------
 * 两版都把 4 组的 8 个 q 字节拼成一条 32B store (整行一次写回), 比 4 次
 * vst1_u8 少 3 条 store; 实测 store 宽度本身不影响带宽, 但条数影响前端.
 *
 * (a) sub+mul 型: t = (v - mn) * inv, 每 8 元素 6 条指令. 精度与旧快路径
 *     完全一致 (减法是 Sterbenz 精确的, 只有乘法有一次舍入).
 * (b) FMA 型:    t = v * inv + bias, bias = -(mn * inv) 预计算,
 *     每 8 元素 5 条指令 —— 省掉一条 FSUB.8H, 实测再快 5~6%.
 *     FMA 的乘积不单独舍入, 所以唯一的新误差来源是 bias 被压进 f16:
 *     |Δq| ≲ |mn*inv| * 2^-11 = 0.125 * |mn|/range 个量化台阶.
 *     mn 相对 range 越大(大 DC 偏置) 抵消越严重, 故由守卫 3 把关. */
static void fc_quant_f16_sub(const float16_t *p, fc_block *out, const float16x8_t mn[4],
                             const float16x8_t inv[4])
{
    for (int r = 0; r < FC_N; r++) {
        const float16_t *row = p + r * FC_N;
        uint8x16x2_t o;
        o.val[0] = vqmovn_high_u16(
            vqmovn_u16(vcvtnq_u16_f16(vmulq_f16(vsubq_f16(vld1q_f16(row + 0), mn[0]), inv[0]))),
            vcvtnq_u16_f16(vmulq_f16(vsubq_f16(vld1q_f16(row + 8), mn[1]), inv[1])));
        o.val[1] = vqmovn_high_u16(
            vqmovn_u16(vcvtnq_u16_f16(vmulq_f16(vsubq_f16(vld1q_f16(row + 16), mn[2]), inv[2]))),
            vcvtnq_u16_f16(vmulq_f16(vsubq_f16(vld1q_f16(row + 24), mn[3]), inv[3])));
        vst1q_u8_x2(out->q + r * FC_N, o);
    }
}

static void fc_quant_f16_fma(const float16_t *p, fc_block *out, const float16x8_t inv[4],
                             const float16x8_t bias[4])
{
    for (int r = 0; r < FC_N; r++) {
        const float16_t *row = p + r * FC_N;
        uint8x16x2_t o;
        o.val[0] = vqmovn_high_u16(vqmovn_u16(vcvtnq_u16_f16(vfmaq_f16(bias[0], vld1q_f16(row + 0), inv[0]))),
                                   vcvtnq_u16_f16(vfmaq_f16(bias[1], vld1q_f16(row + 8), inv[1])));
        o.val[1] = vqmovn_high_u16(vqmovn_u16(vcvtnq_u16_f16(vfmaq_f16(bias[2], vld1q_f16(row + 16), inv[2]))),
                                   vcvtnq_u16_f16(vfmaq_f16(bias[3], vld1q_f16(row + 24), inv[3])));
        vst1q_u8_x2(out->q + r * FC_N, o);
    }
}

/* 守卫 1/2 (两条路径共用):
 * 守卫 1: 把 step==0 换成 65504 后取全部 32 列的最小值, 看 1/step 会不会溢出 f16.
 * 守卫 2: 取全部 32 列 f16 域跨度 mx-mn 的最大值, 看它会不会溢出成 inf.
 * inf 在 vmax 里会可靠地胜出, 所以不需要额外的 isinf 判断 (NaN 输入不支持). */
static inline int fc_guard2(const float16x8_t mn[4], const float16x8_t mx[4],
                            const float16x8_t step16[4])
{
    const float16x8_t big = vdupq_n_f16(FC_F16_MAX);
    const float16x8_t zero16 = vdupq_n_f16(0.0f);
    float16x8_t acc = big, rng = zero16;
    for (int g = 0; g < 4; g++) {
        acc = vminq_f16(acc, vbslq_f16(vceqq_f16(step16[g], zero16), big, step16[g]));
        rng = vmaxq_f16(rng, vsubq_f16(mx[g], mn[g]));
    }
    return (float)vminvq_f16(acc) < FC_FAST_STEP_MIN || (float)vmaxvq_f16(rng) > FC_F16_MAX;
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

    const float16x8_t zero16 = vdupq_n_f16(0.0f);

    if (fc_guard2(mn, mx, step16)) {
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

    /* 守卫 3: bias = -mn*inv 必须能"有意义地"装进 f16, 否则 v*inv 与 bias
     * 相减会把有效位全抵消掉 (q 的整体平移 ≲ |mn*inv|*2^-11 个台阶).
     * 取 |mn*inv| <= 512 —— 等价于 |mn| <= 2.01*range, 此时平移 ≤ 0.25 台阶,
     * 与 sub+mul 版的精度不可区分 (实测两者 ΔSNR 都在 -0.2~-0.3 dB).
     * step==0 (常量列) 时 bias=0, 不触发; mn*inv 溢出成 inf 时必然触发.
     *
     * 注意这里的回退目标是 f16 的 sub+mul 路径 (只慢 ~6%), 不是 f32 路径
     * (慢 2.2x) —— 守卫 1/2 才是块级大惩罚, 守卫 3 只是让这一块从
     * 49 ns 变成 54 ns, 不再制造"一个坏列拖垮整块"的双峰. */
    float16x8_t bias[4], mag = zero16;
    for (int g = 0; g < 4; g++) {
        bias[g] = vnegq_f16(vmulq_f16(mn[g], inv16[g]));
        mag = vmaxq_f16(mag, vabsq_f16(bias[g]));
    }
    if ((float)vmaxvq_f16(mag) > FC_FMA_BIAS_MAX)
        fc_quant_f16_sub(p, out, mn, inv16);
    else
        fc_quant_f16_fma(p, out, inv16, bias);
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

/* ---------- 1/step: f32 除法版 (确定性, 可与标量参考 bit-exact) ----------
 * 快路径用的是 FRECPE (实现定义, 快但不可复现); HQ 路径要能跟标量参考对拍,
 * 所以这里用 vdivq_f32 再窄化 —— 每块只有几十条, 延迟还被后续计算盖住. */
static inline float16x8_t fc_inv_div(const float16x8_t step16, const float32x4_t one)
{
    const float16x8_t zero16 = vdupq_n_f16(0.0f);
    float32x4_t sl = vcvt_f32_f16(vget_low_f16(step16));
    float32x4_t sh = vcvt_high_f32_f16(step16);
    float16x8_t inv = vcombine_f16(vcvt_f16_f32(vdivq_f32(one, sl)),
                                   vcvt_f16_f32(vdivq_f32(one, sh)));
    return vbslq_f16(vceqq_f16(step16, zero16), zero16, inv);
}

/* 候选步长: 把 step0 的 bit pattern 挪 off 个 ulp (step0==0 时保持 0) */
static inline float16x8_t fc_step_shift(const float16x8_t step0, int off)
{
    const float16x8_t zero16 = vdupq_n_f16(0.0f);
    uint16x8_t bits = vreinterpretq_u16_f16(step0);
    bits = (off >= 0) ? vaddq_u16(bits, vdupq_n_u16((uint16_t)off))
                      : vsubq_u16(bits, vdupq_n_u16((uint16_t)(-off)));
    return vbslq_f16(vceqq_f16(step0, zero16), zero16, vreinterpretq_f16_u16(bits));
}

void fc_compress_neon_hq(const fc_f16 *restrict in, fc_block *restrict out)
{
    const float16_t *p = (const float16_t *)in;
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float16x8_t zero16 = vdupq_n_f16(0.0f);

    float16x8_t mn[4], mx[4], step16[4];
    float32x4_t lo_mn[4], hi_mn[4];

    fc_minmax(p, mn, mx);
    for (int g = 0; g < 4; g++)
        step16[g] = fc_step_store(mn, mx, out, g, &lo_mn[g], &hi_mn[g]);

    /* 守卫 1/2 与快路径完全一致: 命中就整块退回 f32 量化 (用现状网格) */
    if (fc_guard2(mn, mx, step16)) {
        float32x4_t lo_inv[4], hi_inv[4];
        for (int g = 0; g < 4; g++) {
            float32x4_t sl = vcvt_f32_f16(vget_low_f16(step16[g]));
            float32x4_t sh = vcvt_high_f32_f16(step16[g]);
            lo_inv[g] = vbslq_f32(vceqq_f32(sl, vdupq_n_f32(0.0f)), vdupq_n_f32(0.0f),
                                  vdivq_f32(one, sl));
            hi_inv[g] = vbslq_f32(vceqq_f32(sh, vdupq_n_f32(0.0f)), vdupq_n_f32(0.0f),
                                  vdivq_f32(one, sh));
        }
        fc_quant_f32(p, out, lo_mn, hi_mn, lo_inv, hi_inv);
        return;
    }

    /* 搜索的求值一律用 sub+mul: 每候选多 1 条指令, 但它在任何 |min*inv| 下都准确.
     * 若改用 FMA 求值, 带大 DC 偏置的块 (|min| 相对 range 很大, 仓库 bench 的
     * pattern 1 就是) 会因加数抵消而把排序算错 —— 所以这条钱不能省. */

    /* ---- 逐组 (8 列) 搜最优网格 ----
     * 残差按 (v-r)*inv 归一化后再平方累加: 直接累加 (v-r)^2 会在小 range 列上
     * 掉进 f16 次正规区甚至归零 (step=2e-5 时 e^2 ~ 1e-10, 远低于 f16 最小次正规
     * 6e-8), 候选之间就没法区分了. 归一化后累加量恒在 O(1), 与数据量级无关. */
    float16x8_t sel_inv[4], sel_bias[4], sel_min[4];
    for (int g = 0; g < 4; g++) {
        float16x8_t best_acc = vdupq_n_f16(FC_F16_MAX);
        float16x8_t best_inv = zero16;
        float16x8_t best_min = mn[g], best_step = step16[g];

        for (int si = 0; si < FC_HQ_NSTEP; si++) {
            float16x8_t stepc = fc_step_shift(step16[g], si - FC_HQ_NSTEP / 2);
            float16x8_t inv = fc_inv_div(stepc, one);
            float32x4_t scl = vcvt_f32_f16(vget_low_f16(stepc));
            float32x4_t sch = vcvt_high_f32_f16(stepc);

            /* 只留 minc 与 acc: 求值已改用 sub+mul, bias 不再进内层循环
             * (少 9 个 live vector, 实测快 ~40%), 选完再算选中候选的 bias */
            float16x8_t minc[FC_HQ_NOFF], acc[FC_HQ_NOFF];
            for (int j = 0; j < FC_HQ_NOFF; j++) {
                float delta = -1.0f + 2.0f * (float)j / (float)(FC_HQ_NOFF - 1);
                float32x4_t t = vmulq_f32(vdupq_n_f32(delta), scl);
                float32x4_t lo = vaddq_f32(lo_mn[g], t);
                t = vmulq_f32(vdupq_n_f32(delta), sch);
                float32x4_t hi = vaddq_f32(hi_mn[g], t);
                minc[j] = vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
                acc[j] = zero16;
            }

            for (int r = 0; r < FC_N; r++) {
                float16x8_t v = vld1q_f16(p + r * FC_N + g * 8);
                for (int j = 0; j < FC_HQ_NOFF; j++) {
                    /* 用 sub+mul 求值 (不用 FMA): 多 1 条指令, 但任何 |min*inv| 下都准 */
                    float16x8_t t = vmulq_f16(vsubq_f16(v, minc[j]), inv);
                    uint16x8_t q = vminq_u16(vcvtnq_u16_f16(t), vdupq_n_u16(255));
                    float16x8_t rr = vfmaq_f16(minc[j], vcvtq_f16_u16(q), stepc);
                    float16x8_t e = vmulq_f16(vsubq_f16(v, rr), inv); /* 归一化残差 */
                    acc[j] = vfmaq_f16(acc[j], e, e);
                }
            }

            /* 归并: 严格小于才更新 (与标量参考的 "if (acc < best)" 同序, 平局取先者) */
            for (int j = 0; j < FC_HQ_NOFF; j++) {
                uint16x8_t better = vcltq_f16(acc[j], best_acc);
                best_acc = vbslq_f16(better, acc[j], best_acc);
                best_inv = vbslq_f16(better, inv, best_inv);
                best_min = vbslq_f16(better, minc[j], best_min);
                best_step = vbslq_f16(better, stepc, best_step);
            }
        }

        sel_inv[g] = best_inv;
        sel_min[g] = best_min;
        sel_bias[g] = vnegq_f16(vmulq_f16(best_min, best_inv)); /* 选中后再算, 与标量参考同式 */
        vst1q_f16((float16_t *)out->min + g * 8, best_min);
        vst1q_f16((float16_t *)out->step + g * 8, best_step);
    }
    /* 最终量化一律用 sub+mul: 与搜索时的求值形式完全一致, 于是"选中的候选"
     * 的实际 SSE == 评估出来的 SSE. 又因为现状网格 (min, step0) 也在候选集里,
     * HQ 的 SSE 严格 <= 现状网格的 SSE —— "永不劣化"是结构性保证, 不是实测巧合.
     * (FMA 那条指令省下来也没意义: 最终量化只占 HQ 的百分之几) */
    fc_quant_f16_sub(p, out, sel_min, sel_inv);
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
