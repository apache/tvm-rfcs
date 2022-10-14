- Feature Name: aarch64_backend
- Start Date: 2022-09-26
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)
- Co-Authors: [@manupak](https://github.com/manupak), [@u99127](https://github.com/u99127)

# Summary

This RFC is to introduce a new TIR backend for AArch64 codegen for supporting target specific features, specifically SVE. Currently AArch64 specific code is generated either through a generic LLVM backend or by tensorize implementation (e.g. the MMLA Arm(R) Neon(TM) instruction), but we could see a benefit from having a more fine grained control over LLVM that targets AArch64.

# Motivation

The main motivation behind this work is to introduce SVE instructions in codegen without changing IRs, scheduling primitives or TVM passes. AArch64 backend would be a good place to work around the issues in LLVM SVE code generation that have surfaced while adding support for SVE in Halide. In addition, `CodegenAArch64` backend would not be limited to SVE codegen – it could be used to introduce AArch64 specific lowering where required, either for specialised use of AArch64 intrinsics or to work around limitations of LLVM.

# Guide-level explanation

In comparison to the Arm(R) Neon(TM) instruction set, which uses a fixed vector length, SVE allows the developer to write vectorized code where the exact length of a vector is unknown at a compile time. That code can then run on hardware implementations with different choices of vector lengths. For hardware implementations, the only constraint for the vector length is that it has to be minimum of 128 bits and it has to be a multiple of 128 bits. *Vscale* is the number of sets of 128 bits that fit into the SVE vector, e.g. vscale of 4 results in a vector length of 512 bits.

The initial SVE implementation in TVM would focus on two main capabilities of SVE:

**1. Vector length agnostic loops**
As an example, consider this vector length agnostic loop that adds two vectors with FP32 elements:

```
for (i = 0; i < n; i += 4 * vscale)
    c[i : i + 4 * vscale] = a[i : i + 4 * vscale] + b[i : i + 4 * vscale]
```

Number 4 in the above example comes from the fact that we can fit four FP32 elements into 128 bits. Here the number of times we have to run the loop will depend on vscale, which is a hardware implementation detail. If the vector length was, as an example, 256 bits, we could process 8 FP32 elements in one iteration, meaning we would have to do `n / 8` iterations. By increasing the vector length to 512 bits, we would need to do `n / 16` iterations.

**2. Predication**
SVE provides support for predication, enabling us to efficiently deal with loop tails, among other things. In the example above, `n` may or may not be a multiple of `4 * vscale`. Predication allows us to handle this loop without any special consideration for the remainder of the elements i.e. `c[n - n % (4 * vscale) : n]`. Essentially, every operation with SVE registers would take a predicate register as one of its arguments that would act as a bit mask indicating which elements are active. Similarly to the vector length, the length of a predicate depends on the hardware implementation.

```
whilelt p0.s, w17, w12
ld1w    { z0.s }, p0/z, [x2, x17, lsl #2]
ld1w    { z1.s }, p0/z, [x1, x17, lsl #2]
fadd    z2.s , z0.s , z1.s
st1w    { z2.s }, p0, [x0, x17, lsl #2]
```

In that example, `whilelt` constructs the predicate register `p0` based on the loop bound variable and the increment variable stored in `w` registers.

## How to target AArch64 backend

Similarly to how we target other LLVM codegen backends, we would invoke AArch64 backend through parsing the `-mtriple` in the target string:

```
target = "llvm -mtriple=aarch64-gnu-linux -mattr=+sve"
```

The node visitors in the AArch64 backend implementation would generate SVE code when `+sve` is part of the `-mattr`.

# Reference-level explanation

The main difference compared to CodegenLLVM would be how we generate LLVM and assembly for `Ramp` and `Broadcast` nodes.

Let's take a simple vectorized addition of two dimensional tensors as an example:

```
A = te.placeholder((200, 200), name="A")
B = te.placeholder((200, 200), name="B")
T = te.compute((200, 200), lambda i, j: A[i, j] + B[i, j])

s = te.create_schedule(T.op)
xo, yo, xi, yi = s[T].tile(T.op.axis[0], T.op.axis[1], x_factor=10, y_factor=5)
                                                                    # ^^ this would be the vector length
s[T].vectorize(yi)
```

Currently, loops that are annotated with vectorize will be represented as `Ramp` nodes in TIR:

```
@main = primfn(A_1: handle, B_1: handle, m: int32, n: int32) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [200, 200], []),
             B: Buffer(B_2: Pointer(float32), float32, [200, 200], [])}
  buffer_map = {A_1: A, B_1: B} {
  realize(compute: Buffer(compute_1: Pointer(float32), float32, [200, 200], []), [0:200, 0:200], True {
    for (i.outer: int32, 0, 20) {
      for (j.outer: int32, 0, 40) {
        for (i.inner: int32, 0, 10) "unroll" {
          compute[(i.inner + (i.outer*10)), ramp((j.outer*5), 1, 5)] = (A[(i.inner + (i.outer*10)), ramp((j.outer*5), 1, 5)] + B[(i.inner + (i.outer*10)), ramp((j.outer*5), 1, 5)])
        }
      }
    }
  })
}
```

The above TIR segment contains static numbers as the lane count (5) and the inferred bound (40) across the `j` axis. If SVE is used, the AArch64 backend would treat the lane count as `llvm.vscale() * 4` and the corresponding loop bound as `ceil( 40 / llvm.vscale() * 4 )`.

With SVE enabled, this TIR would further be lowered to LLVM:

```
for_body7: ; preds = %for_body7, %for_begin4.preheader
  %indvars.iv = phi i64 [ %indvars.iv.next, %for_body7 ], [ 0, %for_begin4.preheader ]
  %14 = trunc i64 %indvars.iv to i32
  %15 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.whilelt.nxv4i1.i32(i32 %14, i32 4)
  %16 = add i32 %12, %14
  %17 = sext i32 %16 to i64
  %18 = getelementptr inbounds float, float* %4, i64 %17
  %19 = tail call <vscale x 4 x float> @llvm.aarch64.sve.ld1.nxv4f32(<vscale x 4 x i1> %15, float* %18)
  %20 = getelementptr inbounds float, float* %5, i64 %17
  %21 = tail call <vscale x 4 x float> @llvm.aarch64.sve.ld1.nxv4f32(<vscale x 4 x i1> %15, float* %20)
  %22 = fadd <vscale x 4 x float> %19, %21
  %23 = getelementptr inbounds float, float* %6, i64 %17
  tail call void @llvm.aarch64.sve.st1.nxv4f32(<vscale x 4 x float> %22, <vscale x 4 x i1> %15, float* %23)
  %indvars.iv.next = add i64 %indvars.iv, %7
  %24 = icmp slt i64 %indvars.iv, 4
  br i1 %24, label %for_body7, label %for_end8, !prof !
```

That would then be turned into assembly when compiled for SVE enabled AArch64:

```
.LBB1_3:
    add w17, w14, w15
    whilelt p0.s, w15, w12
    sxtw    x17, w17
    ld1w    { z0.s }, p0/z, [x2, x17, lsl #2]
    ld1w    { z1.s }, p0/z, [x1, x17, lsl #2]
    add x16, x16, x10
    cmp x16, #4
    add w15, w15, w10
    fadd    z0.s, z0.s, z1.s
    st1w    { z0.s }, p0, [x0, x17, lsl #2]
    b.lt    .LBB1_3
    mov w15, #200
    mov x16, x11
```

## Pattern matching TIR

We can change the fixed vector length loops into scalable vector length loops with following steps:

1. Pattern match vectorized BufferLoad/BufferStore nodes in a form `A[ramp(iter*lanes, 1, lanes)]` and check that they all have the same lane value. Change the lane value to `vscale * 16 / sizeof(dtype)` in generated LLVM.
2. Change the outer loop bound to depend on the `vscale`.
3. If the loop doesn't satisfy this pattern, we abort.

# Drawbacks

Enabling VLA could possibly be done in `CodeGenLLVM` using non-AArch64 specific intrinsics, however, it would be good to improve the confidence in the new features in a specialised backend before introducing them to CodeGenLLVM. Whether the generic VLA intrinsics in LLVM are mature enough is still an open question.

# Prior art

There is already plenty of precedence for specialised LLVM backends, e.g. for Hexagon.
Regarding to SVE support, there has been work happening to add [128 bit SVE vector support to Halide](https://github.com/halide/Halide/pull/6781)
