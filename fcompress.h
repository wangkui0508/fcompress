/* fcompress.h -- 32x32 FP16 分列 8bit 量化压缩 (Apple Silicon / NEON)
 *
 * 数据布局约定 (很重要, 决定了整个 SIMD 结构):
 *
 *   in  : __fp16[32*32], 行主序, in[r*32 + c]
 *   out->q : uint8_t[32*32], 行主序, q[r*32 + c]     <-- 刻意与输入同布局
 *   out->min/step : __fp16[32], 每列一个
 *
 *   q 保持行主序是为了让编解码两端都不需要 32x32 转置:
 *   压缩时按行扫描即可用向量累加器同时维护 32 列的 min/max,
 *   解压时 min[c..c+7] / step[c..c+7] 是连续 8 个 f16, 可直接 vld1q_f16.
 */
#ifndef FCOMPRESS_H
#define FCOMPRESS_H

#include <stddef.h>
#include <stdint.h>

#define FC_N 32
#define FC_ELEMS (FC_N * FC_N)

#if defined(__aarch64__)
typedef __fp16 fc_f16; /* 与 arm_neon.h 的 float16_t 是同一类型 */
#else
typedef _Float16 fc_f16;
#endif

/* _Alignas(16) 让 min/step 也落在 16B 边界上 (偏移 1024 / 1088),
 * 于是 vld1q_f16 全部是对齐访存 (AArch64 不要求, 但更省事) */
typedef struct {
    _Alignas(16) uint8_t q[FC_ELEMS]; /* 量化值, 行主序 */
    fc_f16 min[FC_N];                 /* 每列最小值 (f16, 精确) */
    fc_f16 step[FC_N];                /* 每列步长 (f16), 反量化: v = min + step * q */
} fc_block;

/* 这个布局就是落盘/传输格式: 不允许出现 padding, 偏移也不允许漂移 */
_Static_assert(sizeof(fc_block) == 1152, "fc_block 必须是 1152 字节的紧凑布局");
_Static_assert(offsetof(fc_block, min) == 1024 && offsetof(fc_block, step) == 1088,
               "min/step 偏移变了, 落盘格式不再兼容");
_Static_assert(_Alignof(fc_block) == 16, "fc_block 需要 16B 对齐以支持 vld1q_f16");

/* 编码: 32*32*2B = 2048B -> 1024B(q) + 128B(min/step) = 1152B
 *
 * 前置条件 (两条编码路径都适用): in 的 1024 个元素必须全部是有限值.
 * NaN/±Inf 不支持 —— NEON 的 FMIN/FMAX 会把 NaN 当缺失值跳过, 而标量参考在
 * 首元素为 NaN 时会污染整列, 两者行为不一致, min/step 与 q 都不保证有意义.
 *
 * 默认路径: 量化在 f16 域完成, 用一条 FMLA.8H 算 t = v*inv + bias
 * (bias = -min*inv 预计算), 再 FCVTNU.8H 转 u16 —— 每 8 元素 5 条指令.
 * 比 f32 路径快约 2.2x, q 与参考实现最多差 1 个 LSB.
 * 以下情况会自动回退 (只有 q 变精确, min/step 不受影响):
 *   a) 某列 step 小到 1/step 会溢出 f16 (range < ~5.1e-3)   -> f32 量化, 约 2.2x 慢;
 *   b) 某列跨度 mx-min 超过 f16 最大值 (range > ~65504), 否则 v-min 在 f16 域
 *      溢出成 inf, 该列顶部整段被饱和到 q=255                 -> f32 量化, 约 2.2x 慢;
 *   c) 某列 |min| > 2*range: FMLA 的两个加数几乎等大, 抵消会吃掉有效位
 *      (q 整体平移最多 |min*inv|*2^-11 个台阶)                -> f16 的 (v-min)*inv
 *                                                              两指令版, 只慢约 6%.
 * 回退是块级的: 32 列里只要有 1 列命中, 整块都走该回退路径. a/b 的代价是
 * 2.2x, c 的代价只有 6%, 所以真正的"双峰"仍然来自 a/b (近似常量列、超量程列).
 * 另外, 本路径的倒数用了实现定义的 FRECPE.8H, 因此只保证重建质量,
 * 不保证字节流跨芯片/工具链一致. */
void fc_compress_neon(const fc_f16 *restrict in, fc_block *restrict out);

/* 高精度路径: 量化在 f32 域, 输出与 fc_compress_ref 逐位一致, 且不依赖任何
 * 实现定义的估计指令, 跨机器字节流稳定.
 * 需要可复现的确定性编码 (落盘、校验、上游比对) 时用它 (代价: 编码慢约 1.9x). */
void fc_compress_neon_hp(const fc_f16 *restrict in, fc_block *restrict out);

/* 解码: min/step 取 f16, q 拓宽到 f16 后用 FMA 一次成型 */
void fc_decompress_neon(const fc_block *restrict in, fc_f16 *restrict out);

/* 标量参考实现, 用于 bit-exact 对拍 */
void fc_compress_ref(const fc_f16 *restrict in, fc_block *restrict out);
void fc_decompress_ref(const fc_block *restrict in, fc_f16 *restrict out);

#endif /* FCOMPRESS_H */
