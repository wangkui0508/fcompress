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
	rm -f $(OBJS) test_fcompress bench_variants
