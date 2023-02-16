- Feature Name: Introduce PresburgerSet
- Start Date: 2023-02-13
- RFC PR: https://github.com/apache/tvm-rfcs/pull/99/
- GitHub Issue: https://github.com/apache/tvm/issues/14006

# Summary
It would be great if TVMScript can grow into a generic programming language in marchine learning domain. To reach that, it seems some powerful analysis tools are needed. Integer set is pivotal in IR analyzing, but IntSet in TVM only represents ranges. This RFC is to seek an improvement for it so that we can perform the IR analysis more precisely. We found the Presburger Set in MLIR library could be leveraged for this purpose.
# Motivation
Current dependence analysis is carried out roughly and inconveniently. Due to the absence of necessary basic infrastructure, it's hard to consider complex if conditions in IR, and it's even more challenging to do element-wise dependence analysis, both for inside or between TIR block analysis.
## Inner block analysis
One goal of TVMscript is to provide an easy & flexible way to construct computation workload for both TVM compiler developers and machine learning algorithm developers. TVMscript requires users to annotate some extra information when programming, such as `T.block`/`T.remap`/`T.init`..., which seem not very intuitive if developers do not have deep compiler knowledge. If IR analysis can help users annotate this kind of information automatically, it would be a tremendous programming option for many cases. All the above analysis requires a better data dependency analysis, which needs a more sophisticated integer set utility. If we can analyze the element-wise dependency between different loop instances, it should be easy to detect the spatial/reduce iteration axis automatically. Take the following loop statement for example, how to easily analyze whether axis `i` is spatial or not, without element-wise dependency analysis?
```
for i, j, k, m, n in grid(37, 23, 40, 57, 60):
    if 3*m + 7n < 58 and 45*k + 77*j >= 34:
        B[i*3324 + j*23 + k*103 + m*279] = A[i, j, k, m, n]
```
If we can detect the reduce axis, `T.init` pattern detection should not be a problem, then. Of course, this auto-detection should only work as an option because TVM developers may sometimes still want to handcraft some complex blocks, such as block nesting.
## Interblock analysis
Most TIR primitives and passes need to analyze the data denpendency between producer & consumer blocks. Without sufficient utility, it's hard to consider complex `if` conditions. Analysis without if-conditions leads to a rough dependency result and causes redundant data transfer & computation. The redundant workload could be neglective for CPU/GPU, but it could be painful for NPU, for which extra data is devastating both for DMA and computation. An if-condition-aware integer set should solve this problem. And even if there is no `if` condition in blocks, `T.Read`/`T.Write` is needed when constructing the workload, which also can be easily inferred from IR stmt if a better IntSet exists. Here is an example:
```
@T.prim_func
def if_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (60,), "float32")
    C = T.match_buffer(c, (20, 20), "float32")
    with T.block():
        B = T.alloc_buffer((60,), dtype="float32")
        for i in T.grid(60,):
            with T.block():
                B[i] = A[i]
        for i, j in T.grid(20, 20):
            with T.block():
                if i + j < 20 and i - j <= 0:
                    C[i, j] = B[i + 2*j]
                else:
                    C[i, j] = 0.0
```
How to determine the maximum range of `B` and do compact buffer range for it? Shape of `B` only needs (39,) and currently CompactBufferAllocation does nothing on this. If we can determine the range automatically, the `T.Read`/`T.Write` annotations could be saved .
# Guide-level explanation
The proposal is to implement a `PresburgerSet` class. The key point is to support inequation constraints to consider if-conditions, so the inequation on multiple Vars will mainly be used to express the sets. The basic set manipulation functions and constructor functions in `IntSet` class will be reimplemented in `PresburgerSet`. An additional constructor function will be added to construct from inequations:
```
PresburgerSet FromConstraints(Array<Constraint> inequations)
```
In order to manage the analysis, we need to separate the vars in all the inequations into at least two kinds, the iteration(or domain) vars and other vars, say local(target) vars. `PresburgerSet` keeps the relationship between iteration vars and local vars, from iteration to local vars or vice versa. Some other utility functions are needed to transform the relationship, including:
```
PresburgerSet reverse()
```
  Reverse the relationship from local vars to iteration vars. So we can further analyze dependency based on read/write sets.
```
PresburgerSet apply_iteration()/apply_local()
```
  Merge two relationships targeting the iteration vars or local vars. Then we can propagate the relationship between multiple sets.
```
PresburgerSet solve_bounds(PrimExpr expr)
```
  Prove engine to solve the maximum/minimum optimization problem based on the inequations in `PresburgerSet`, such as simplex solver. Input parameter expr is the target expression of optimization problem.

Other existing API, like intersect/union etc, will be reimplemented based on updated data structure.
# Reference-level explanation
You may have already noticed that this is just what other modern integer set library provides, like ISL. So an economical way to achieve this is to leverage existing public wheels. ISL is mostly used, but it seems not modular enough or open enough, so it could be difficult to integrate deeply. Presburger Set located in MLIR is modular designed and open developed. So building from it would be a good choice.

No need to introduce MLIR as a source code submodule. Installing LLVM prebuilt package installs the necessary libs of Presburger Set, so MLIR can be integrated into TVM just like LLVM codegen uses LLVM libs, and it can be switched on/off on demand. The new-added util function needs to check whether MLIR is installed when called and falls back to the interval set when MLIR is not found.
# Drawbacks
The `PresburgerSet` serves as the basic infrastructure of IR analysis, and a wide range of lowering passes/primitives may need it. Part of its functionality is duplicated with `IntSet`, so people should make a decision which one to use according to the analysis task.
# Alternatives
The other way is to handcraft a copy of code similar to Presburger Set in MLIR, which minimizes the software dependence of TVM project, but it needs considerable effort and seems like reinventing the wheel, if there is no extra new idea to implement.
# Future possiblities
One day, when we make sure `PresburgerSet` can fully cover what `IntSet` provides, in terms of functionality and efficiency, we may consider phasing out the legacy IntSet, then no more decisions about `PresburgerSet` and `IntSet`.
