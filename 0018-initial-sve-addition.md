- Feature Name: Adding Initial SVE Support to TVM 
- Start Date: 2021-07-30
- RFC PR: https://github.com/apache/tvm-rfcs/pull/18

Authors: Meera Nakrani, Sjoerd Meijer

## Introduction

In this RFC we would like to propose a TIR extension to support scalable
vectorisation. Scalable vectorisation is extracting data parallelism from 
code, but as opposed to a fixed width vectorisation, the vector length is 
unknown at compile time. A scalable vector's total number of elements is a 
constant multiple of a specified number of elements. The 
[LLVM LangRef](https://llvm.org/docs/LangRef.html) refers to this constant 
multiple as vscale. It is a positive integer that is unknown at compile time, 
therefore the overall vector length (VL) is also unknown. The value of vscale, 
and therefore VL, will depend on the architecture that is running the program. 
More details and an overview of this is given in 
[this tutorial](https://www.stonybrook.edu/commcms/ookami/support/_docs/ARM_SVE_tutorial.pdf), 
where an example of a daxpy kernel is given from slide 17 onwards. In this RFC, 
we will show an example of lowering from TE for a (scalable) vector addition 
kernel all the way down to LLVM IR, further illustrating the vscale concept. 
We will also cover TIR support and how it affects the LLVM codegen. This is an 
introductory RFC to see if the design of our prototype implementation, see 
https://github.com/apache/tvm/pull/8655, is sound and we welcome any feedback 
on this prosposal.

Before we explain this in more detail, let's first briefly look at the current
state and terminology with an example. Vectorisation along the x-axis of an
addition of two one-dimensional tensors A and B of size 18, writing the result
to C, will result in the following TIR:

```
C[ramp(0, 1, 17)] = A[ramp(0, 1, 17)] + B[ramp(0, 1, 17)]`
```
where the Ramp TIR node has the form 'Ramp(base, stride, lanes)' showing that
these elements are processed in (vector) lanes.

The size of 18 has been chosen to demonstrate the challenges of vectorising
this example. Vector architecture extensions (e.g. X86 AVX512 or AArch Neon)
typically allow to pack and operate on a power-of-2 number of elements, so 2,
4, 8, 16, etc.  elements. If the elements are integers, and a vector register
is 128-bits wide, we can pack 4 integer elements into one vector register (if
an integer is 4 bytes). This is an example of fixed width vectorisation,
because the vector registers have a fixed width of 128-bits. Since we have 18, the
number of elements in the vectors A, B, and C, is not a multiple of 4, we need
4 vector operations processing 4 * 4 = 16 elements, and 2 scalar operations are
required for processing the 16th and 17th elements which we call the scalar
epilogue.

## Motivation

However, most modern vector architectures (e.g. X86 AVX512 and the Arm
Architecture's MVE and SVE extensions) support predicated vector instructions,
removing the need for such a scalar epilogue and also allowing more code to be
vectorised.  Lane predication allows the enabling/disabling of certain lanes in
vector operations.  This allows us to have just 5 vector operations for our
example, and importantly no scalar epilogue. But since we do not need to
process 5 * 4 = 20 elements, the last vector operation only needs to write two
elements, which can be achieved by predication as we can enable the first two
lanes and disable the last 2 lanes.

In addition to predication, and also related to it, some new vector 
architectures also allow scalable vectorisation. As opposed to so called fixed
width vectorisation (e.g. AArch Neon), the Arm architecture SVE vector
extension allows implementations to choose a vector register length between 128
and 2048 bits.  It supports a vector length agnostic programming model which
allows code to run and scale automatically across all vector lengths without
recompilation.

## Problem Statement

We would like to add support for Arm Architecture's Scalable Vector Extension (SVE) 
in TVM by introducing features for Vector Length Agnostic (VLA) programs and
predication, i.e. the 2 main new SVE features. Thus we would like to express
scalable vectorisation in both TE and TIR. The question is how to achieve that? In
Tensor Expression language, our example to add two tensors A and B would look
like this:

```
n = 17
A = te.placeholder((n,), name="A", dtype = "int8")
B = te.placeholder((n,), name="B", dtype = "int8")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
s = te.create_schedule(C.op)
x, = C.op.axis
s[C].vectorize(x)
```

Vectorisation along the x-axis is requested with `vectorize(x)`, and will
result in the TIR example shown in the Introduction. However, this requires
knowing the vector length at compile time; it is an example of fixed width
vectorisation. Instead, we would like for it to work with an unknown vector
length at compile time.

## Solution Approach

In order to address the problem of expressing scalable vectorisation, we would
like to propose the addition of a new `vectorize_scalable` function to the Tensor
Expression language, for example:
``` 
s[C].vectorize_scalable(x)
```
The TIR output of this would be:

```
primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {C: Buffer(C_2: Pointer(int8), int8, [17], []),
             A: Buffer(A_2: Pointer(int8), int8, [17], []),
             B: Buffer(B_2: Pointer(int8), int8, [17], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (i: int32, 0, 17;i+=VL) {
    C_2[ramp(i, 1, VL)] = ((int8xVL*)A_2[ramp(i, 1, VL)] + (int8xVL*)B_2[ramp(i, 1, VL)])
  }
}
```

In the above TIR, we can see the the for loop is looping with an agnostic
stride `VL`, which stands for Vector Length. `VL` is only showed for ease of
representation and we don't store `VL` anywhere inside the TIR data structures.

We can also see the syntax of the Ramp nodes have now been modified to handle
an unknown vector length, as seen by `ramp(i, 1, VL)`, instead of a fixed
integer. The form is still `Ramp(base, stride, lanes)` and the semantics of it
are still the same, the only difference is that the number of lanes is unknown
at compile time, and so we use VL as a way of representing that. For fixed-width 
vectorization, the number of lanes is treated as a single input whereas in scalable 
vectorization, VL represents a vector-length wide input segment. 


## Implementation 

An agnostic constructor has been added to the Ramp node, as well as to the
Broadcast node, with an additional parameter. This parameter is a boolean named
`is_scalable`, in order to enable both fixed and scalable vectorisation.

This boolean has also been added in `data_type.h` as the type of the Ramp node
has changed, it is now scalable. The constructor is:

```
DataType(int code, int bits, int lanes, bool is_scalable = false)
```

Originally, for fixed vectorisation, `is_scalable` will be false, but when
scalable vectorisation is enabled we will set `is_scalable` to true.

In TIR we introduced a new ForKind called `kVectorizeScalable` which marks a 
loop as able to be vectorized but the value of VL will be unknown. This loop 
is then legalised during a new pass called `VectorizeLoopScalable` pass, which 
is triggered by the `vectorize_scalable` function mentioned previously. This pass 
transforms the loop so that it is able to handle the unknown constant VL. Our 
prototype was implemented before the addition of the While node to TVM, and so 
it currently transforms a For loop into a variable For loop. To do this, the For 
node had to have extra parameters added to its implementation that would only be 
used in this one specific case. One change we are planning to make is to make use 
of this existing While node and transform a For Loop into a While loop during the 
`VectorizeLoopScalable` pass, since it is the more natural choice and it is what 
will be generated in the assembly as well. Once the loop has been legalised, it 
is passed to the code generator which will translate it to the relevant LLVM SVE 
instrinsics. The intrinsics involved are for predicated load/stores, loop 
increments and predication mask calculation. While predication is not implemented 
as a TIR construct, it is used to support VLA. 

This is all handled on the TE and TIR end, but for code generation we introduce
a file called `codegen_aarch64.cc` which handles the creation of SVE intrinsics
in LLVM by visiting the Load, Store and For nodes (to be replaced with a While
node) and generating the relevant LLVM IR. For the above TIR output, the expected 
LLVM IR would be:
```
for_body:                                         ; preds = %for_body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for_body ], [ 0, %entry ]
  %8 = trunc i64 %indvars.iv to i32
  %9 = tail call <vscale x 4 x i1> @llvm.aarch64.sve.whilelt.nxv4i1.i32(i32 %8, i32 17)
  %10 = getelementptr inbounds float, float* %4, i64 %indvars.iv
  %11 = tail call <vscale x 4 x float> @llvm.aarch64.sve.ld1.nxv4f32(<vscale x 4 x i1> %9, float* %10)
  %12 = getelementptr inbounds float, float* %5, i64 %indvars.iv
  %13 = tail call <vscale x 4 x float> @llvm.aarch64.sve.ld1.nxv4f32(<vscale x 4 x i1> %9, float* %12)
  %14 = fadd <vscale x 4 x float> %11, %13
  %15 = getelementptr inbounds float, float* %6, i64 %indvars.iv
  tail call void @llvm.aarch64.sve.st1.nxv4f32(<vscale x 4 x float> %14, <vscale x 4 x i1> %9, float* %15)
  %indvars.iv.next = add i64 %indvars.iv, %7
  %16 = icmp slt i64 %indvars.iv, 17
  br i1 %16, label %for_body, label %for_end, !prof !5
```
In the IR above we have: 
* the `whilelt` target specific SVE intrinsic, which generates a mask 
(variable %9) based on the induction variable (%8) and the loopbound (i32 17). 
This mask is only consumed by load/store instructions/intrinsics (and not e.g. 
data processing instructions such as the fadd), which is how predication is 
currently modeled in LLVM.
* `<vscale x 4 x float>` is a scalable vector of 4 float elements
* `<vscale x 4 x i1>` is the scalable mask consisting of 4 boolean values.


## Drawbacks
1. We've not been able to reuse existing code for this prototype, instead we have
a whole a new vectorization pass and a new code generator for AArch64 specific 
intrinsics.
2. In addition to this, we have created a new scheduling primitive which will need 
support and maintenance. This also means that the autotuner and autoscheduler must
be modified to include the new rules needed for this primitive. These new rules are
not supported in this prototype but have been noted as extension work in the future. 

## Next Steps 

As this is a prototype, we are aware that there are still areas that need
further work:
1. Our prototype was written before the While node work was merged. Instead of
   creating a separate variable For-loop we should be using a While node, as
   mentioned above. We think this is a straightforward rewrite, and with this
   rewrite completed we think that this work lays the non-user facing part and
   foundation for scalable vectorisation.
2. As discussed in the RFC, the prototype currently has a new function 
   `vectorize_scalable` to trigger scalable vectorisation. In response to the 
   comments on the RFC, we will need to ammend this so that we use the existing
   `vectorize` function but with the additional boolean parameter `scalable`, which
   will default to false, but will be set to true to enable scalable vectorisation. 
3. The user facing part includes work to support the auto-scheduler and
   auto-tuner will also need to be added. This would enable us to generate a
   schedule that will call `vectorise_scalable` and thus perform scalable
   vectorisation, allowing users to benefit from this addition.
4. This prototype also provides a few more opportunities for development, for
   example implementing variable loops when we come across nested for loops. On
   top of this, SVE support can continue to be extended by implementing other
   features such as Gather-load and Scatter-store patterns.

## Acknowledgements
The prototype is based on earlier work by Giuseppe Rossini.
