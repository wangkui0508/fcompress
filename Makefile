CC      ?= clang
# Apple M1/M2/M3/M4/M5 都支持 FP16 向量算术; 用 -mcpu=apple-m1 作为安全下界,
# 也可以写 -march=armv8.6-a+fp16
ARCH    ?= -mcpu=apple-m1
CFLAGS  ?= -O3 -std=gnu11 $(ARCH) -Wall -Wextra -Wno-unused-function
LDFLAGS ?= -lm

OBJS = fcompress_neon.o fcompress_ref.o test_fcompress.o

test_fcompress: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

# 回答"fdiv 能不能换成取倒数"的对照实验 (9 个变体)
bench_variants: bench_variants.c fcompress_neon.c fcompress.h
	$(CC) $(CFLAGS) -I. -o $@ bench_variants.c fcompress_neon.c $(LDFLAGS)

variants: bench_variants
	./bench_variants

# 快路径优化前后 A/B + 收益归因 (OLD/NEW/U4/FMA/HP + pass1/pass2 诊断)
exp_opt: exp_opt.c fcompress_neon.c fcompress_ref.c fcompress.h
	$(CC) $(CFLAGS) -I. -o $@ exp_opt.c fcompress_neon.c fcompress_ref.c $(LDFLAGS)

# HQ 编码器: bit-exact 对拍 + 精度增益 + 编码耗时
exp_hq: exp_hq.c fcompress_neon.c fcompress_ref.c fcompress.h
	$(CC) $(CFLAGS) -I. -o $@ exp_hq.c fcompress_neon.c fcompress_ref.c $(LDFLAGS)

.PHONY: hq
hq: exp_hq
	./exp_hq

# 访存宽度微基准: 证明这个 kernel 是访存受限而不是浮点受限
exp_ld: exp_ld.c fcompress.h
	$(CC) $(CFLAGS) -I. -o $@ exp_ld.c $(LDFLAGS)

.PHONY: opt
opt: exp_opt
	./exp_opt

%.o: %.c fcompress.h
	$(CC) $(CFLAGS) -c -o $@ $<

# 打印是否拿到 FP16 向量算术 + 生成的 NEON 指令
.PHONY: check-arch asm
check-arch:
	@printf '#include <arm_neon.h>\n' > /tmp/_fc_probe.c
	@$(CC) $(CFLAGS) -dM -E /tmp/_fc_probe.c | grep -E '__ARM_FEATURE_FP16_(VECTOR_)?ARITHMETIC|__ARM_NEON' | sort

asm: fcompress_neon.c
	$(CC) $(CFLAGS) -S -o - fcompress_neon.c | grep -E '^\s+(fmin|fmax|fmla|fmul|fsub|fdiv|ucvtf|fcvtn|fcvtl|xtn|shrn|ld1|st1)' | sort | uniq -c | sort -rn

clean:
	rm -f $(OBJS) test_fcompress bench_variants exp_opt exp_ld exp_hq
