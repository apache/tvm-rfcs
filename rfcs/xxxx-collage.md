# Design Doc: Collage [Draft 0.7]

```
Feature Name: Collage
Start Date: Mar 2022
Authors: Mark Shields (mbs@octoml.ai)
RFC PR: <tbd>
GitHub Issue: <tbd>
```

This design doc (with accompanying
['v2' prototype implementation](https://github.com/mbs-octoml/mbs-tvm/tree/mbs-collage-sketch))
shows how to bring tuning to TVM's operator fusion and BYOC partitioning passes. The tuning search explores the choice
of sub-graphs (aka 'partitions') as well as choice of toolchain (TVM native or one of the available BYOC integrations,
aka 'backends') for each candidate kernel so as to minimize the expected model inference latency. We call the result
an 'optimal partitioning'. This new tuning layer complements the tuning traditionally done by TVM and other toolchains
during lowering. It can also complement any global tuning, for example to explore all possible global layouts.

The approach is based on the [preprint](https://arxiv.org/pdf/2111.00655.pdf):

> *Collage: Automated Integration of Deep Learning Backends*  
> Byungsoo Jeon, Sunghyun Park, Peiyuan Liao, Sheng Xu, Tianqi Chen, Zhihao Jia

This tuning approach contrasts with TVM's existing "greedy" and "manual" approaches to fusion and BYOC:

- Greedy: Currently only the largest possible supported sub-graphs are used for kernels, irrespective of their execution
  time. With Collage many more candidate sub-graphs are explored, and it is possible for two smaller sub-graphs to yield
  better overall latency than one large sub-graph if they mix toolchains.
- Manual: Currently the TVM user must commit to a BYOC toolchain and invoke the corresponding partitioning function
  before the main TVM compilation flow proceeds. With Collage the choice of toolchain can be automated based on measured
  latency. Collage will also explore mixing and matching between multiple BYOC toolchains as well as TVM's native
  backend.

The design (when Collage is enabled) subsumes TVM's fixed `FuseOps` and BYOC-provided `partition_for_<toolchain>`
operations (built using the `MergeComposite`/`AnnotateTarget`/`MergeCompilerRegions`/`PartitionGraph` passes) with a
single new
`CollageFuseOps` pass. The pass is carefully engineered to build directly on the existing `"TOpPattern"` attributes
(provided for every Relay operator and used by `FuseOps`), BYOC `"target.<toolchain>"`
operator predicates (provided for some operator/toolchain pairs by 'operator-based' BYOC integrations) and BYOC operator
pattern/predicates (registered in the pattern table by 'pattern-based' BYOC integrations). In this way only the more
boilerplate aspects of existing BYOC integrations need to be adjusted to support Collage. The
`partition_for_<toolchain>` operations are retained for users who wish to retain manual control.

> NOTE: We'd like to coordinate these changes with the UMA project. Our aim in this design is to make the smallest
> changes to BYOC as possible. We think the changes described here can be easily reworked to follow any BYOC API
> proposals settled on by UMA. See also "Related Work."

Collage offers four advantages:

- **Latency**: Overall model latency may be reduced compared to TVM native, TVM with a specific BYOC toolchain, or a
  non-TVM compiler such as TensorRT.
- **Automation**: The choice of which BYOC toolchains to enable can be automated.
- **Economy of implementation**: Five standalone passes using three separate mechanisms for expressing fusion
  rules/algorithms and implementing partitioning can be replaced with one, which itself is built from compositional
  primitives.
- **Decoupling**: It is ok for a candidate kernel found during search to actually not be valid for a toolchain (even
  TVM's). Such candidates could be given 'infinite' cost and thus ignored during search. In this way we can avoid tight
  coupling between backends and fusion rules.

## FAQ

Pending.

## Success Metrics

1. Collage offers at least a 10% latency improvement for a selection of standard ONNX models and NVIDIA hardware using
   targets which include the CuDNN and CuBlas libraries, the CUTLASS library (with tuning, via BYOC), the TensorRT
   compiler (via BYOC), and (obviously!) TVM native.
2. Collage does not require new per-target or per-model patterns or rules to be implemented independently of the BYOC
   integrations.
3. Collage with just the native TWM and a single BYOC toolchain enabled is never worse than using the
   existing `partition_for_<toolchain` method in TVM today.

## Project Milestones

- [Done] M0: Port paper prototype to recent TVM main and validate paper results.
- [Done] M1: Internal design doc.
- [Done] M2: Use 'v2' prototype to test design doc, and rework ready for TVM community.
- [In progress] M3: RFC
- [2022Q1] M4: Re-validate results on 'v2' prototype for larger models (eg GPT2) and more NVIDIA targets.
- [2022Q2] M5: Implementation in TVM main, including 'sub-projects' listed below.
- [OctoML internal] M6: Estimator integrated into OctoML platform, validation against OctoML test suite.
- [OctoML internal] M7: Productionization for OctoML.

## Check-in plan

Though the 'v2' prototype is in a personal branch we'd like to transition to main ASAP and rely on directory/namespace
separation, maintaining backwards compat, and a new `PassConfig` flag to isolate all Collage changes from the rest of
TVM. A rough PR progression is:

- TensorRT and CUTLASS BYOC changes are backwards compat. The existing `partition_for_X` functions remain. The
  CUTLASS-specific tuning and codegen functions will either continue to be supported or we'll work with users to account
  for them being folded into the function-at-a-time `relay.ext.cutlass`
  codegen function.
- The the `DFPattern` and friends changes are all mostly just for improving the robustness of the
  `IndexedGraph<T>` class and can go into main independently.
- Some basic `Expr` improvements can go into main independently.
- The design allows for multiple `Target`s for the same `DLDeviceType`. That requires the various
  `build` interfaces which currently accept `Union[Target,Dict]` to also accept a list of `Target`s, and can be
  backwards compat.
- The new Collage code can go in bottom-up as we develop unit tests:
    - Support utils, including `NameSupply`, `IndexSet`, `PriorityQueue`, `Cost`, `CostEstimator`.
    - The core `SubGraph` datatype.
    - `CandidateKernel`.
    - The `FusionRule` class hierachy (which itself can be broken into sub-PRs).
    - `FusionSpec`.
    - `GatherFusionSpecs` helper for bridging the existing BYOC world with the Collage 'FusionRule' world.
    - The `CollageFuseOps` driver pass itself.

## Related Work

- The [Cascading Scheduler](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0037-arm-ethosu-cascading-scheduler.md) combines i) dynamic-programming
  to find an optimal grouping of TE sub-expressions, ii) an analytic model of cost to guide the search,
  and iii) cascading scheduling of the TE sub-expressions so as to reduce memory high-watermark. By contrast
  Collage i) also uses dynamic-programming, but to find an optimal grouping of Relay sub-expressions, ii)
  uses measurement to guide the search and iii) assuming the toolchain will 'do its best' with the
  sub-graph offered to it.
- The [Universal modular Accelerator Interface](https://github.com/apache/tvm-rfcs/pull/60) proposal
  adds a layer on top of the existing and separate TVM BYOC, operator strategy, operator scheduling,
  target-specific passes and target-specific code generation extension points. Collage currently relies
  only on the global pattern registry and global `relay.ext.<toolchain>` function to integrate with BYOC
  integrations, but this is trivial to change should this project change the source of truth.

## Example

We start with `mod` bound to [MNIST](https://github.com/onnx/models/tree/main/vision/classification/mnist):

```
fn (%x: Tensor[(1, 1, 28, 28), float32]) -> Tensor[(1, 10), float32] {
  %0 = nn.pad(%x, 0f, pad_width=[[0, 0], [0, 0], [2, 2], [2, 2]]);
  %1 = nn.conv2d(%0, meta[relay.Constant][0] /*Tensor[(8, 1, 5, 5), float32]*/,
                 padding=[0, 0, 0, 0], channels=8, kernel_size=[5, 5]);
  %2 = add(%1, meta[relay.Constant][1] /*Tensor[(8, 1, 1), float32]*/);
  %3 = nn.relu(%2);
  %4 = nn.max_pool2d(%3, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0]);
  %5 = nn.pad(%4, 0f, pad_width=[[0, 0], [0, 0], [2, 2], [2, 2]]);
  %6 = nn.conv2d(%5, meta[relay.Constant][2] /*Tensor[(16, 8, 5, 5), float32]*/,
                 padding=[0, 0, 0, 0], channels=16, kernel_size=[5, 5]);
  %7 = add(%6, meta[relay.Constant][3] /*Tensor[(16, 1, 1), float32]*/);
  %8 = nn.relu(%7);
  %9 = nn.max_pool2d(%8, pool_size=[3, 3], strides=[3, 3], padding=[0, 0, 0, 0]);
  %10 = reshape(%9, newshape=[1, 256]);
  %11 = nn.dense(%10, meta[relay.Constant][4] /*Tensor[(10, 256), float32]*/, units=None, out_dtype="float32");
  add(%11, meta[relay.Constant][5] /*Tensor[(1, 10), float32]*/)
}
```

We can compile this with Collage enabled for a variety of NVIDIA toolchains/libraries as follows:

```
with tvm.transform.PassContext(config={"relay.fallback_device_type": 2, "relay.collage.enable_collage": True}):
    host_target = tvm.target.Target("llvm")
    generic_target = tvm.target.Target("cuda", host_target)
    cutlass_target = tvm.target.Target("cuda -compiler=cutlass", host_target)
    tensorrt_target = tvm.target.Target("cuda -compiler=tensorrt", host_target)
    cudnn_target = tvm.target.Target("cuda -libs=cudnn", host_target)
    cublas_target = tvm.target.Target("cuda -libs=cublas", host_target)
    targets = [generic_target, cutlass_target, tensorrt_target, cudnn_target, cublas_target]
    exe = tvm.relay.vm.compile(mod, target=targets)
```

(Note that `cudnn` and `cublas` are not yet supported in the 'v2' prototype.)

After the `CollageFuseOps` pass, the intermediate `"main"` global function could resemble the following (though we've
modified this "optimal" partitioning by hand to illustrate all the varieties of kernels so don't take it as
representative of actual performance):

```
fn (%x: Tensor[(1, 1, 28, 28), float32]) -> Tensor[(1, 10), float32] {
  # Use TVM native
  %3 = fn (%FunctionVar_08: Tensor[(1, 1, 28, 28), float32],
           Primitive=1) -> Tensor[(1, 1, 32, 32), float32] {
    nn.pad(%FunctionVar_08, 0f, pad_width=[[0, 0], [0, 0], [2, 2], [2, 2]])
  };
  %4 = %3(%x);
  # Use TVM native, but indicate we wish to link to CuDNN
  %6 = fn (%FunctionVar_07: Tensor[(1, 1, 32, 32), float32],
           Primitive=1) -> Tensor[(1, 8, 28, 28), float32] {
    %5 = fn (%FunctionVar_5: Tensor[(1, 1, 32, 32), float32],
             Composite="cudnn.conv2d") -> Tensor[(1, 8, 28, 28), float32] {
      nn.conv2d(%FunctionVar_5, meta[relay.Constant][0] /*Tensor[(8, 1, 5, 5), float32]*/,
                padding=[0, 0, 0, 0], channels=8, kernel_size=[5, 5])
    };
    %5(%FunctionVar_07)  
  };
  %7 = %6(%4);
  # Use TVM native, with fusion
  %8 = fn (%FunctionVar_06: Tensor[(1, 8, 28, 28), float32],
           %FunctionVar_12: Tensor[(8, 1, 1), float32],
           Primitive=1) -> Tensor[(1, 8, 28, 28), float32] {
    %3 = add(%FunctionVar_06, %FunctionVar_12);
    nn.relu(%3)
  };
  %9 = %8(%7, meta[relay.Constant][1] /*Tensor[(8, 1, 1), float32]*/);
  # Use TVM native
  %10 = fn (%FunctionVar_05: Tensor[(1, 8, 28, 28), float32],
            Primitive=1) -> Tensor[(1, 8, 14, 14), float32] {
    nn.max_pool2d(%FunctionVar_05, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
  };
  %11 = %10(%9);
  # Use TVM native
  %12 = fn (%FunctionVar_04: Tensor[(1, 8, 14, 14), float32],
            Primitive=1) -> Tensor[(1, 8, 18, 18), float32] {
    nn.pad(%FunctionVar_04, 0f, pad_width=[[0, 0], [0, 0], [2, 2], [2, 2]])
  };
  %13 = %12(%11);
  # Use TensorRT, with fusion
  %14 = fn (%FunctionVar_03: Tensor[(1, 8, 18, 18), float32],
            %FunctionVar_11: Tensor[(16, 1, 1), float32],
            Primitive=1,
            Compiler="tensorrt",
            global_symbol="collage_nn_conv2d_add_nn_relu_1") -> Tensor[(1, 16, 14, 14), float32] {
    %1 = nn.conv2d(%FunctionVar_03, meta[relay.Constant][2] /*Tensor[(16, 8, 5, 5), float32]*/,
                   padding=[0, 0, 0, 0], channels=16, kernel_size=[5, 5]);
    %2 = add(%1, %FunctionVar_11);
    nn.relu(%2)
  };
  %15 = %14(%13, meta[relay.Constant][3] /*Tensor[(16, 1, 1), float32]*/);
  # Use TVM native
  %16 = fn (%FunctionVar_02: Tensor[(1, 16, 14, 14), float32],
            Primitive=1) -> Tensor[(1, 16, 4, 4), float32] {
    nn.max_pool2d(%FunctionVar_02, pool_size=[3, 3], strides=[3, 3], padding=[0, 0, 0, 0])
  };
  %17 = %16(%15);
  # Use TVM native
  %18 = fn (%FunctionVar_01: Tensor[(1, 16, 4, 4), float32],
            Primitive=1) -> Tensor[(1, 256), float32] {
    reshape(%FunctionVar_01, newshape=[1, 256])
  };
  %19 = %18(%17);
  # Use CUTLASS, with fusion
  %20 = fn (%FunctionVar_0: Tensor[(1, 256), float32],
            %FunctionVar_1: Tensor[(10, 256), float32],
            %FunctionVar_2: Tensor[(1, 10), float32],
            Primitive=1,
            Compiler="cutlass",
            global_symbol="collage_cutlass_dense_bias_nn_dense_add") -> Tensor[(1, 10), float32] {
    %1 = fn (%FunctionVar_01: Tensor[(1, 256), float32],
             %FunctionVar_11: Tensor[(10, 256), float32],
             %FunctionVar_21: Tensor[(1, 10), float32],
             Composite="cutlass.dense_bias") -> Tensor[(1, 10), float32] {
      %0 = nn.dense(%FunctionVar_01, %FunctionVar_11, units=None, out_dtype="float32");
      add(%0, %FunctionVar_21)
    };
    %1(%FunctionVar_0, %FunctionVar_1, %FunctionVar_2)
  };
  %20(%19, meta[relay.Constant][4] /*Tensor[(10, 256), float32]*/, meta[relay.Constant][5] /*Tensor[(1, 10), float32]*/)
}
```

## Design

The implementation is mostly under `src/relay/collage/...` (namespace `tvm::relay::collage`), with some helper Python
under `python/tvm/relay/collage`.

If the `relay.collage.enable_collage` `PassConfig` attribute is true then a new `CollageFuseOps` pass is inserted before
the existing `FuseOps` pass. The new pass effects the invariant:

> All Relay sub-graphs in all global functions which are to be lowered to a kernel are replaced by calls to an inline
> `"Primitive"` `Function`. Functions which are to be lowered by a BYOC-provided toolchain are given
> `"Compiler"` and `"global_symbol"` attributes. The bodies of those function may contain calls to inlined
> `"Composite"` annotated functions to further direct lowering within the kernel.

The `CollageFuseOps` pass proceeds in four phases:

- **Phase 1**: The available `Target`s are scanned to build a list of `FusionSpec`s. Each `FusionSpec` is built from
  (a tree of) `FusionRule`s. How the rules are constructed depends on `Target` itself. The remaining phases execute on
  each global function separately.
- **Phase 2**: A `DataflowGraph` is constructed for the global function. The available `FusionRule`s are evaluated on
  the dataflow graph to yield a (possibly overlapping) set of `CandidateKernels` for each target. Each candidate is
  described by a `SubGraph` which efficiently denotes a sub-graph of the global function's body without the need to
  construct any new expressions. The candidates are placed in a `CandidateKernelIndex` for use below.
- **Phase 3**: A shortest path is found in the following (implicit) search graph:
    - Search Nodes: An `IndexSet` describing which dataflow nodes are been assigned to a candidate kernel so far.
    - Search Edge X->Y: A `CandidateKernel` can be applied to node X to give node Y. The candidate is disjoint from all
      dataflow nodes already assigned in X. To avoid an unnecessary search space explosion the candidate must also
      include the next yet-to-be-assigned dataflow node in X.
    - Edge cost: Estimated latency of the candidate kernel, plus a kernel launch penalty. Note that though we need to be
      able to extract the candidate's sub-graph in order to build the kernel, we do not yet need to partition the
      overall function body expression.
  Other search algorithms are certainly possible, eg the Paper uses an evolutionary search to refine
  the partitioning found by the dynamic-programming search. We can easily abstract away the search
  interface to support multiple implementations in the future.
- **Phase 4**: The function body is partitioned according to the candidate kernels on the shortest path.

In the following we introduce the new datatypes, then expand on the phases.

### Util Datatypes

- `PostDfsIndex`: The integer index of a Relay sub-expression in a post-dfs traversal of the overall Relay expression.
  If index i is less than index j then we know the sub-expression for j cannot influence the value of the sub-expression
  for i.
- `DataflowGraph`: As alias for the existing `IndexedGraph<Expr>` from the `DFPatternMatcher` suite (which in turn is a
  reworked copy of the `IndexedGraph` private to `fuse_ops.cc`). It is used throughout to manage the three-way bijection
  from Relay `ExprNode`s to `PostDfsIndex`s to
  `DataflowGraph::Node`s. Each `DataflowGraph::Node` describes the sub-expression's dataflow inputs, outputs, dominator
  and inverse-dominators.
- `IndexSet`:  A bit vector indexed by `PostDfsIndex`s. These are used as a compact representation for an arbitrary set
  of dataflow nodes in a dataflow graph.
- `Cost`: A `double` representing a candidate kernel's 'cost', which currently is just mean execution latency in
  seconds. Collage only cares that costs are additive and a total order, so in the future we could support cost
  functions which balance execution time against high memory watermark or other measures. Costs may be `Unknown`
  (ie NaN) to signal some other heuristic should be used to compare kernel costs. Costs may be `Invalid` (ie +inf)
  to signal the toolchain could not compile and run a candidate kernel.

### SubGraph

A `SubGraph` is an `IndexSet` of the `PostDfsIndex`s of all dataflow nodes 'inside' an arbitrary sub-graph of the
overall dataflow graph. This and `FusionRule` below are the core Collage datatypes.

Sub-graphs can be used to represent 'composite'' and 'fused' functions without having to pay the cost of constructing
either the function or the rewritten overall 'partitioned' expression which calls that function. We also allow functions
to be extracted independently of partitioning, since we'll need to estimate the latency of many more kernel functions
than will ultimately be used in the final Relay expression. We expect O(thousands) of sub-graphs to be in flight while
processing a given model.

A sub-graph classifies every dataflow node of the overall expression as either 'inside' or 'outside' the sub-graph.
Obviously not all such divisions make sense, for example it is not valid for an inside node to feed into another inside
node via outside nodes. We provide the `IsValid` method to check for validity, and `SubGraphConfig` to control which
rules apply (such as maximum depth).

As well as 'inside' and 'outside' we have four other flavors of dataflow nodes (all uniquely determined from the
'inside' nodes):

- 'entry' nodes are those inside with at least one dataflow input outside.
- 'exit' nodes are those inside with at least one dataflow output outside, or which are considered 'external' in the
  underlying dataflow graph (eg because they represent the result of the overall function).
- 'input' nodes are those outside with at least one dataflow output inside.
- 'output' nodes are those outside with at least one dataflow input inside.

It is valid to have multiple entry nodes (we'll bind a parameter for each). It may be valid to have multiple exit
nodes (we'll build a tuple of all such). It may be valid to have exit nodes which also contribute to other inside
nodes (ie represent a 'top' on an intermediate result).

Sub-graphs are closed under:

- Disjoint union.
- Wrapping by a label, which indicates the wrapped sub-graph should be extracted as a sub-function with a "Composite"
  label.
- Substitution, which allows a sub-graph w.r.t. one dataflow graph to be transformed to match some other (typically
  smaller) dataflow graph.

To support some of the `OpPatternKind`-based fusion rules (see below) we give sub-graphs a kind, which is generally the
maximum of the kinds of all the operator calls appearing inside it. We also given sub-graphs a label to help debugging.

Note that the Relay `PatternPartitoner` goes directly from `Expr` to partitioned `Expr` without stopping at any
intermediate representation. It may be worth 'promoting' `SubGraph` out of Collage andy into the standard `DFPattern`
suite.

Note that to support closure on both disjoint union and wrapping by a label `SubGraph`s are actually recursive -- see
the 'v2' prototype `sub_graph.cc` for details.

### CandidateKernel

A `CandidateKernel` pairs a `SubGraph` with a `FusionSpec` (from which the intended `Target` for the candidate kernel
can be extracted). All Collage search and measurement is in units of candidate kernels.

### FusionRule

A `FusionRule` describes how to find a set of `CandidateKernel`s for a `DataflowGraph`. This and `SubGraph` above are
the core Collage datatypes. All fusion rules implement the method:

```
virtual Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                                   const FusionSpec& spec) const;
```

The candidates are allowed to overlap, and ultimately it is the job of the Collage fusion searcher to find a selection
of candidates which covers the whole Relay expression without overlap.

We provide a set of 'base' fusion rules which produce candidates from the dataflow graph directly. We also provide a set
of 'combinator' rules which can produce new candidates from the results of arbitrary sub-rule or sub-rules. In this way
it is possible to combine the fusion rules to express a wide variety of fusion strategies, akin to the way we can
combine TVM passes.

There may be many thousands of candidates in flight during the fusion search. We take care to defer rewriting any Relay
expressions (eg to extract the fused function, or partition the model) until absolutely necessary.

The base rules implemented so far:

- `DFPatternFusionRule`: Given a `DFPattern` and expression predicate, produces a candidate for every sub-graph matched
  by the pattern and predicate. Unlike the Relay `PatternRewriter`, candidates are free to overlap. This is the
  foundation for pattern-based BYOC integrations, and can be used to write targeted fusion rules as well as find
  examples of 'composite' operators.
- `OpPredicateFusionRule`: Given an attribute name, produces a candidate for every call to a primitive Relay operator
  where the operator has predicate bound to that attribute which returns true given the call sub-expression. Generally
  this will result in a singleton sub-graph containing only the call, but it may pull in constant arguments to the call
  should they be required. This is the foundation for operator-based BYOC integrations, though we should consider
  retiring this mechanism in favor of pattern-based alone.
- `OpCallByKindFusionRule`: Uses the `"TOpPattern"` attribute provided for every Relay operator to produce a candidate
  for every call to a 'fusable Relay operator'. This can be used as the foundation for generic fusion patterns which
  work over all Relay operators with particular properties (elementwise, broadcast, injective, reductive, anchor).

The combinator rules implemented so far:

- `CompositeFusionRule`: 'Tags' the candidates matched by an arbitrary sub-rule with the rule name. Tagged sub-graphs
  are turned into "Primitive" Function with the "Composite"
  attribute bound to the tag. This can be used to indicate Relay operators (or groups of Relay operators) are to be
  rewritten to specific target-specific operators. This combinator wraps the `DFPatternFusionRules` for the
  pattern-based BYOC integrations. However it could also be used with the default TVM backend, eg to indicate Relay
  operators should be replaced with particular external library implementations.
- `CombineByPrimitivesFusionRule`: Given a sub-rule and a list of 'primitive' rules, finds all possible ways of
  combining the sub-rule candidates to yield even larger candidates. Note that the sub-rule's candidates may also be
  included in the results -- that is every combination of candidates is considered optional. The 'primitive' rules allow
  combining by
  `OpPatternKinds`, and combining the arguments to tuples which themselves are arguments to Relay operator calls. This
  rule is intended to mimic the existing TVM `FuseOps` pass, though: i) all combinations are found, ii) the starting set
  of candidates can be provided by any other rule (ie not just `OpCallByKindFusionRule`), and iii) we rely on `SubGraph`
  validity checking to weed out infeasible candidates.

Though not yet implemented, we'd like to allow a combinator rule which will union candidate based on their 'anchor'
operators. This can be used to implement 'vertical' and 'horizontal' fusion on more primitive candidates. Note that the
`SubGraph` machinery supports multiple-input and -output sub-graphs and their validation, so horizontal fusion is easy
implement.

We also have `MaxCoalesceFusionRule`, which eagerly combines 'touching' candidates (ie candidates where the output of
one sub-graph can be directly connected to the input of the other sub-graph)
to form the largest possible candidate. The idea is once the search has been completed this rule can be used to collapse
adjacent kernels intended for the same target.

Here's some typical `FusionRule` combinations for different fusion strategies (please excuse the crudity of the diagram,
I didn't have time to build it to scale or paint it):

- Classic TVM `FuseOps`:

```
      OpCallByKindFusionRule
                |
                v
    CombineByPrimitivesFusionRule (with default TVM primitive rules)
```

- Classic operator-based BYOC with `AnnotateTarget`/`MergeCompilerRegions`/`PartitionGraph` passes:

```
      OpPredicateFusionRule
                |
                v
   CombineByPrimitivesFusionRule (with join anything primitive rule)
```

- Classic pattern-based BYOC with `MergeComposite`/`AnnotateTarget`/`PartitionGraph` passes:

```
     DFPatternFusionRule(pattern1)  ...  DFPatternFusionRule(patternn)
                |                                    |
                v                                    v
      CompositeFusionRule(label1)   ...   CompositeFusionRule(labeln)
                        \                     /
                         v                   v
                            UnionFusionRule
                                   |
                                   v
                     CombineByPrimitivesFusionRule (with join anything primitive rule)
```

- "Just fuse what I tell you to fuse", using `DFPatterns` to directly select candidates:

```
     DFPatternFusionRule(pattern1)  ...  DFPatternFusionRule(patternn)
                           \                 /
                            v               v
                             UnionFusionRule
```

- "Consider this library implementation for these sub-expressions", using `DFPatterns` to pick out which Relay operators
  are supported (note that TVM lowering does not currently support this):

```
    OpCallByKindFusionRule     DFPatternFusionRule(pattern1) ... DFPatternFusionRule(patternn)
                    \                       |                                 |
                     \                      v                                 v
                      \         CompositeFusionRule(label1)  ...  CompositeFusionRule(labeln)
                       \                    |                        /
                        v                   v                       v
                                      UnionFusionRule
                                            |
                                            v
                             CombineByPrimitivesFusionRule (with default TVM primitive rules)
```

### FusionSpec

A `FusionSpec` pairs a a `FusionRule` with a `Target`.

### Phase 1

We build on the existing TVM support for heterogeneous devices and targets. The available `Targets` are extracted from
the compilation configuration (eg using the existing `CompilationConfig` helper class). Each target is inspected to
decide on how to construct a `FusionSpec`, which will guide Collage in the selection of candidate kernels to explore for
that target.

- If the `Target` has a `"fusion_spec"` attribute, use that directly (not currently in the 'v2' prototype). This would
  allow users to directly control fusion for the target's they care about.
- If the `Target` has a `"compiler"` attribute (eg `"cutlass"`), and the global pattern table has an entry for that
  attribute value, assume the `Target` denotes a pattern-based BYOC integration to explore. The `FusionSpec`
  will import all the BYOC patterns and predicates automatically.
- As above, but if global pattern has no matching entry, assume the `Target` denotes a predicate-based BYOC integration
  to explore (eg `"tensorrt"`). The `FusionSpec` will look for and evaluate predicates with the
  `"target.<compiler>"` attribute on all Relay operators.
- Otherwise, assume the `Target` denotes a TVM-native target. The `FusionSpec` mimics the existing `FuseOps`, but now
  generalized to explore multiple candidates so as to leave room for possible BYOC candidates.

Note that to make this approach work we need to allow for multiple `Target`s with the same `DLDeviceKind`. For the VM
simply switching the `target` argument from dictionary to list and removing some redundant Python preprocessing code was
all that was required to support this.

The user can use `on_device` annotations to constrain sub-graphs to particular devices. When Collage is considering
candidate kernels, it should be sure to choose a candidate `Target` which 'refines' the `Target` for every
sub-expression discovered by the `PlanDevicesPass`. Given targets T and U we say 'T refines U' if T has a
'"compiler"' and/or '"fusion_spec"' attributes, U has no such attributes, and T and U otherwise agree on all other
fields. (This is not currently in the 'v2' prototype).

### Phase 2

Most of the hard work for this phase is carried by the `AllCandidateKernels` implementations of the `FusionRule`s. The
main driver simply needs to index all the found `CandidateKernels` by their minimum 'inside' `PostDfsIndex`
for rapid retrieval during the shortest path search.

### Phase 3

We find it most natural to use Dijkstra to find the optimal partitioning. A `SearchState` is:

- An `IndexSet` of the dataflow nodes already 'covered' by candidates on the best path to this state. This is the
  identifying key for the state.
- The predecessor `SearchState` in the best path to this state.
- The `Cost` of the best path to this state. This is the order for the Dijkstra priority queue.
- The `CandidateKernel` for the transition from the best predecessor to this state.

The starting state has no covered nodes. The final state has all nodes covered.

When expanding a state we could choose any `CandidateKernel` collected from phase 2 provided it doesn't overlap with the
state's covered set. However, a search path applying candidates C then D is equivalent to one applying D then C, so we
only consider candidates which intersect the next yet-to-be-covered dataflow node. For each such candidate we use
the `CostEstimator` (with it's assumed cache) to get the candidate's cost, build the successor state, and 'relax' the
successor state in the usual way.

Not all Relay expression nodes need to be assigned to a kernel since the VM or other execution provider can happily
evaluate most Relay expressions except for calls to primitive operators. Thus the search must allow for the possibility
of a expression node being 'left behind'.

### Phase 4

The overall Relay expression is partitioned over all the `CandidateKernel`s on the shortest path 'in parallel'. Since
all the candidates are expressed using `SubGraph`s w.r.t. the original dataflow graph, we must be careful not to
invalidate yet-to-be-partitioned candidates as we go. Working backwards in dataflow order avoids this problem.

Note that all the extracted functions in the result will be marked as `"Primitive"`, and thus will be left alone by most
other Relay passes except `LowerTEPass`. Thus it's fine for `FuseOps` to be run (repeatably) after
`CollageFuseOps`.

## Known Limitations

- **Some BYOC boilerplate changes required**: TVM's current BYOC integration API only requires the 'lowering/codegen'
  function to be registered to a well-known global function name. Everything else is up to the BYOC author.
    - Collage requires pattern-based BYOC integrations to register their patterns in the global pattern table.
    - Collage requires the BYOC lowering function to yield a valid `runtime::Module` without requiring any additional
      BYOC-specific passes to be run.
    - Collage requires the BYOC integration to either correctly test for which operators are supported in the
      pattern/operator predicate, or gracefully propagate failure rather than CHECK-fail if an unsupported operator is
      included in a candidate kernel. Thus a BYOC integration will need to be 'robustified' to become 'Collage
      compatible'. Overall we've tried to make as few changes as possible. Collage will happily follow along with any
      improvements to the BYOC integration API (eg via the UMA project).
- **Higher tuning cost**: Obviously Collage needs to estimate the latency of many more candidate kernels, and each
  candidate may itself trigger tuning during lowering. For TVM this can require O(thousands) of trials and take O(hours)
  , so we'll be very dependent on cached tuning logs to amortize this cost between models for the same target.
  Currently Collage will measure more candidates even if TVM native is the only available target.
- **Task extraction vs Tuning**: Traditionally TVM has had three phases: i) Task extraction (find the fused sub-graphs
  to tune), ii) Tuning (find a good schedule for those sub-graphs), and iii) Compilation (re-compile the model, now
  retrieving schedules for all the anticipated sub-graphs from the cache.) However the Collage 'v2' prototype collapses
  all these phases. This lets us lazily explore the implied search graph (nodes = partially rewritten models, edges =
  selected of sub-graph and toolchain as a candidate kernel, cost = estimated sum of kernel costs plus launch penalties)
  , and thus only pay the cost of tuning candidate kernels which could possibly influence the final partitioning.
- **No non-local optimization**: Though Collage can explore the choice of sub-graph and toolchain, it cannot explore any
  choices which require the arguments and/or result of the sub-graph to be rewritten. Thus Collage **cannot** be used to
  search over:
    - choice of layout for arguments/results (may require insertion of layout transforms),
    - choice of memory scope for arguments/results (may require insertion of device copies),
    - choice of device on which to host the kernel (ditto)
      since all those choices can require changes beyond the candidates sub-graph.
- the choice of layout for a kernel since any choice other than the model's default must be
  'corrected' for by the inserted layout transformations. To support this efficiently we'd need to abandon the
  simple-minded but fast `SubGraph` representation we describe below in favor of something like an EGraph
  representation, which seems like a very large change for TVM.
- **Dependency management**: Currently BYOC integrations tend to assume they are the only non-TVM toolchain in use. So
  it's possible two toolchains introduce runtime dependencies which can't be satisfied. Collage has no notion of
  dependencies or incompatibilities and may attemt to mix candidate kernels we can't support in prod. It's also possible
  for two BYOC integrations to have incompatible runtimes.
- **Additive kernel cost assumption**: Collage as per this design assumes the cost of running candidate kernels is
  additive, plus a small launch penalty. However cache effects can dominate measured latency, particularly for 'light'
  kernels. Thus there may be a **additive error** in the final result:

  > additive_error = measured_latency(collage_partitioning) - sum_{kernel} (estimated_latency(kernel) + penalty)

  The evolutionary search explored by the Collage paper can help here since it uses measured end-to-end model latency as
  its cost function, but we're deferring that to future work.

- **Limited search space**: Naively exploring all sub-graphs is O(n!), so we need to constrain the search. The easiest
  approach is just to limit candidate kernels to sub-graphs of just a few operators. This can mean significatly faster
  candidates are not explored, yielding a partitioning with high **optimality loss**:

  > optimality_loss = measured_latency(collage_partitioning) - measured_latency(true_optimal_partitioning)

  Though the 'true' optimal partitioning may be infeasible to find, the user may easily discover a high
  **apparent loss**, eg by comparing the Collage result with a traditional BYOC partitioning result:

  > apparent_loss = measured_latency(collage_partitioning) - measured_latency(users_own_partitioning)

- **Fragile toolchains**: Some BYOC toolchains are intended to be stand-alone compilers in their own right, and have
  been tuned against common models and include global flags to guide optimizations such as reducing precision. However
  Collage will only feed these toolchains smaller sub-graphs, thus making the limited search space problem more severe.
- **High variance in lightweight kernels**: Small kernels can have high variance, thus the choice of which toolchain to
  use can be arbitrary. We probably want to i) validate our variance estimator is accurate, ii) choose a percentile
  slightly above 50% for the estimated candidate kernel latency, and iii) fall back to hard-coded priorities when the
  measured variance is too high.
- **Non-compositional BYOC toolchains**: BYOC partitioning functions often run global passes to get the Relay graph into
  a state better aligned with the toolchain on the assumption they are the exclusive partitioning pass. Most obvious is
  the choice of layout, and if two BYOC integrations have a different choice of layout then there's currently no way for
  them to be used concurrently. All of those passes must either be i) pushed up to global configuration (which could be
  explored by a search layer outside of TVM), ii) pushed into the BYOC lowering/codegen function (to prepare the
  sub-graph for further compilation) or iii) moved into the standard Relay optimization passes run before
  `CollageFuseOps`.
- **Repeated FuseOps**: Some passes (eg `ManifestAlloc`) introduce new calls to primitive function which must be fused
  and lowered, even though the main work of fusion and lowering has already occurred. We'll need to either
  retain `FuseOps`, or ensure `CollageFuseOps` retains the efficiency and handling of `FuseOps` when there's no
  toolchain ambiguity.
- **Explainability**: It's easy to show the user the final partitioning and estimated times for each kernel, but harder
  to show why that partitioning won out over all others during search.
- **Does not subsume `partition_for_<toolchain>`**: We don't have any plan to deprecate the existing patterns of each
  BYOC integration supplying a `partiion_for_<toolchain>` function. If the user has a specific toolchain in mind then
  making the partition explicit enjoys both faster compilation and can incorporate global optimization passes which
  Collage cannot currently account for (eg enforcing a particular layout).

## Sub-projects

These items need more design and can be run as 'sub-projects'.

### Robust candidate kernel latency measurement

Collage requires an implementation of a `CostEstimator`:

```
class CostEstimator {
 public:
  /*!
   * \brief Return the estimated cost (possibly after many many minutes of training time) of
   * running function using target.
   */
  virtual Cost Estimate(const Function& function, const Target& target) const;
}
```

The 'v2' prototype has implemented this with an in-memory cache and a small Python driver which defers to
TVM's `tvm.runtime.vm.VirtualMachine`s `benchmark` helper. The following needs to be designed and implemented:

- Compilation should be in units of `IRModule` rather than `Function` so that, in the future, additional global
  definitions (such as for weights) can be conveyed to the toolchain.
- The recent MetaSchedule work has provided `BuilderInput` (`include/tvm/meta_schedule/builder.h`),
  `RunnerInput` (`include/tvm/meta_schedule/runner.h`) and `Database` (`include/tvm/meta_schedule/database.h`)
  interfaces. The latter is for `TuningRecord`s of `Workload`s. It looks like these interfaces can support the
  measurement of Collage `CandidateKernel`s with minor changes.
- (Internal to OctoML) We need an implementation connecting to the internal OctoML kernel tuning workflow and production
  cache. Ideally this would be the same implementation as for the MetaSchedule system.
- Collage converts measured 50th %ile latencies to costs in seconds. We may need to consider taking a slightly higher
  %ile to be more robust against variance on small kernels. We need to validate the estimated variance reflects true
  variance.
- For TVM-native targets, we would like the `Estimate` call to perform any TVM tuning required for a novel candidate
  kernel.

### Easier Library Integration

TVM has two very different ways to make external library implementations available for use by kernels: The pattern-based
BYOC approach and the TVM `te.extern` approach.

The pattern-based approach allows library implementations to match with more than one Relay operator, such as for biased
convolution with an activation function. For example, for
[DNNL](https://oneapi-src.github.io/oneDNN/v1.3/index.html) the global pattern table is extended
in `python/tvm/relay/op/contrib/dnnl.py`, and the pattern labels indicate the intended corresponding DNNL functions. The
user is responsible for partitioning using the usual `MergeComposite`/`AnnotateTarget`/`PartitionGraph`
sequence. The `relay.ext.dnnl` BYOC function in `src/relay/backend/contrib/dnnl/codegen.cc` looks for calls to
`"Composite"` functions in the overall `"Primitive"` function, and dispatches based on the `"Composite"` label. C code
is emitted to target the DNNL library, and the standard C compiler helper is invoked to produce a
`runtime::Module`.

Note that it is not possible for a TVM-generated kernel to call a library function integrated this way. In effect every
library function must go into a library-specific kernel (though kernels may group calls to multiple library function).

The `te.extern` approach only allows library implementations which are 1:1 with Relay operators. However the library may
be used as part of a larger TVM-generated kernel, and the usual TVM tuning machinery may choose to use the library based
on overall kernel performance measured during TVM tuning. For example, `batch_matmul`
can be implemented using [CuBLAS](https://developer.nvidia.com/cublas) via the strategy `batch_matmul` in
`python/tvm/contrib/cublas.py`, which is made available to the operator's `OpStrategy` using
`batch_matmul_stategy_cuda` in `python/tvm/relay/op/strategy/cuda.py` when `cublas` appears in the `Target`s `libs`
attribute. That strategy simply calls the `PackedFunc` registered as `tvm.contrib.cublas.batch_matmul` and implemented
in `src/runtime/contrib/cublas/cublas.cc` as part of the TVM runtime.

Collage as presented can work with either approach. For the pattern-based BYOC approach Collage doesn't need to know
what's going on under the BYOC integration hood, it only needs to see a `Target` with the appropriate
`compiler` attribute. For the `te.extern` approach Collage can choose a candidate TVM sub-graph, then rely on TVM tuning
to redirect some operators to their library implementations should the `Target` have the appropritae `libs`
attribute.

However, better would be something which:

- Supports the many-to-one mapping of the pattern-based approach since it is so common in library implementations.
- Always allows calls to extern functions from within TVM-generated kernels.
- Requires less boilerplate than the pattern-based approach, and less ceremony than the `te.extern` approach.

Our strawman:

- Allow calls to `"Composite"` Functions to be transliterated to extern calls in the normal TVM lowering flow, where
  the `"Composite"` attribute gives us the 'external function label'.
- The transliteration uses a global TVM registry of external function labels. Each entry describes how to generate a
  library shim and how to emit a `tir.call_packed' to that shim.
- The usual Collage fusion rules can be used to include labelled sub-graphs with the appropriate external function
  labels as alternatives. Those sub-graphs are ultimately combined into candidate kernels. Collage will then naturally
  search between candidates with different choices of native vs library implementations.

### Robust BYOC integrations for targets of interest

Overall any BYOC toolchain which could be supported by Collage needs to be brought to a high standard:

- It should support the latest toolchain/library versions.
- It should support as much of Relay (both operators and dtypes) as feasible. In particular, Collage will only find
  interesting mixes when BYOC toolchains have overlapping operator and dtype support.
- It should correctly report which operators/patterns are supported.
- It should have good unit test coverage in CI.
- Dependencies should be documented and installation scripted (hopefully this is an easy consequence of the above).
- The translation scheme should give the BYOC toolchain the best chance to do well. In particular, if Collage reports
  toolchain X 'is better' than toolchain Y for a candidate sub-graph we want to have confidence that's not just because
  toolchain Y has been hobbled by a poor translation, API misuse, or other 'holding it wrong' issue.
- Where feasible, partitioning for the BYOC toolchain (not using Collage) should not be worse than using the toolchain
  directly.

Our current focus is on TensorRT, CUTLASS, CuDnn and CuBlas.

### Visualization

A [netron](https://netron.app/) style visualization for Relay which clearly shows the partitioning and cost for all the
kernels would be very valuable. The paper prototype produces such a visualization but we've lost that functionality in
the transition to 'v2'.

## Highlights from the 'v1' prototype

The results of the preprint were derived in a [branch](https://github.com/cmu-catalyst/collage) from
[TVM](https://github.com/apache/tvm) at `461d06eb5cfc7954f1983779acd05c47cea269f1`. We ported/rebased that code onto
main, and refer to it as the
['v1' prototype implementation](https://github.com/mbs-octoml/mbs-tvm/tree/mbs-collage-port).

The 'v1' prototype has five main parts:

- A
  new [backend](https://github.com/mbs-octoml/mbs-tvm/blob/52d8780e879a9115b8a93e505bcd3a6c2646c61f/include/tvm/ir/expr.h#L208)
  field on every Relay `Expr` to capture the pattern name and backend name chosen by Collage to force compilation to
  match its choices.
- An [intercept](https://github.com/mbs-octoml/mbs-tvm/blob/52d8780e879a9115b8a93e505bcd3a6c2646c61f/src/relay/transforms/fuse_ops.cc#L1392)
  in `fuse_ops.cc` which redirects to the main Collage fuser/searcher before TVM’s fusion rules kick in.

- The main
  fuser/searcher [implementation](https://github.com/mbs-octoml/mbs-tvm/blob/52d8780e879a9115b8a93e505bcd3a6c2646c61f/python/collage/optimizer/comp_graph_optimizer.py#L221)
  (for the simpler DP algorithm). This implementation:
    - Uses both Relay `Pattern` s and it’s own path-based fusion algorithm to find candidate sub-graphs.
    - Uses the DP algorithm to find the best assignment of fused sub-graphs and targets to cover the whole Relay graph.
    - Applies the assignment to the IRModule using the new `backend` field

  The evolutionary search algorithm runs after the above and attempts to replace ‘op’ kernels (use a library) with
  ‘graph’ kernels (if there’s a unique graph backend).
- An
  intercept ([here](https://github.com/mbs-octoml/mbs-tvm/blob/52d8780e879a9115b8a93e505bcd3a6c2646c61f/src/relay/transforms/fuse_ops.cc#L1402)
  and
  [here](https://github.com/mbs-octoml/mbs-tvm/blob/52d8780e879a9115b8a93e505bcd3a6c2646c61f/python/collage/optimizer/_optimizer.py#L48))
  in `fuse_ops.cc` to actually effect the fusion for BYOC backends depending on the new `backend` field
- An
  intercept ([here](https://github.com/mbs-octoml/mbs-tvm/blob/52d8780e879a9115b8a93e505bcd3a6c2646c61f/src/relay/backend/te_compiler_cache.cc#L284)
  and
  [here](https://github.com/mbs-octoml/mbs-tvm/blob/52d8780e879a9115b8a93e505bcd3a6c2646c61f/python/collage/backend/collage_strategies.py#L18))
  in `te_compiler.cc` to take over the selection of `OpStrategy` based on the `backend` field.

Note that the 'v1' prototype only supports `IRModules` with a single `"main"` whose body is in the ‘pure dataflow’ Relay
subset. Ie only calls, tuples, tuple projections, function variables and constants are supported.

## Differences between the Paper's prototype and this Design

In comparison to the 'v1' prototype, this design:

- Avoids the need to add any new 'Collage specific' fusion patterns and predicates. We want to make sure Collage can
  work even for out-of-tree BYOC toolchains (modulo some of the BYOC API changes we discuss below).
- Builds on the existing support for heterogeneous `Target`s to represent the menu of available toolchains to use during
  search. In particular, we want to allow users to blend `on_device` annotations (to express preferences for which
  devices should execute which sub-graphs) with Collage (to find the best kernels and toolchains respecting those device
  preferences).
- Uses the existing convention for `"Primitive"`, `"Composite"` and `"Compiler"` attributes on Relay `Function`s to
  express the assignment of sub-graph to toolchain.
- Implements support for 3rd party libraries (eg cudnn) so as to allow an N-to-1 mapping from Relay operators to library
  call (this is not yet implemented in the 'v2' prototype, see below for the sketch).
- Is implemented mostly in C++.

However:

- The 'v2' prototype only implements the 'op-level' dynamic-programming based search strategy from the paper. Though the
  paper reports encouraging results with the 'graph-level' evolutionary-search strategy we leave that to future work.

## TODO in the 'v2' prototype

- Implement extern-for-TVM support and bring in `cudnn` and `cublas`.
- Cross-check against one of the 'v1' models.
- Bring up on `GPT2`.
- Explore `float16` performance mixing `CUTLASS` and `TensorRT`.
- Implement TVM-tuning during Collage search.
- Connect estimator to production tuner & cache.
- Estimator works on `IRModule` not `Function`. Resolve `params` binding question.
- Find model+target combination that shows compelling speedup from mixing w.r.t. all other options, including stand
  alone `TensorRT`.
- Implement Target refinement so that device planning can be used to constrain the available Collage targets to consider
  for arbitrary sub-graphs. Allow multiple targets per `FusionSpec` so that we don't waste time finding the same
  candidates for different TVM targets.
- 'Lookahead' from the current search state to find the 'next' dataflow node(s) which have candidates crossing multiple
  `FusionSpec`s. That defines a sub-graph. There's no need to search over all possible candidates within that sub-graph
  since almost certainly the maximal candidates will be best. Somehow prune the candidates to implement that.
- Cleanup after search to merge adjacent kernels for the same target when supported by toolchain.
- How much of the existing `DFPattern` machinery should be refactored to go via `SubGraph`?
- Post fusion passes introduce new Relay primitives which then need to be fused and lowered, so `FuseOps` still in pass
  list. Consider replacing with `CollageFuseOps` in lightweight mode? Need to avoid all search when toolchain is already
  uniquely determined.
- `Target`s can have a `"fusion_spec"` attribute to directly control fusion.
- Indexing in `CombineByKindFusionRule` to avoid O(n^2) iteration over candidates.
- Need to be dominator aware in `CombineByPrimitivesFusionRule` or is current naive approach of
  using `SubGraph::IsValid`
  good enough to eliminate taps?
- What's with the use of `OpPatternKinds` on dataflow edges in `FuseOps` and the special rule relabelling
  `kBroadcast` as `kElemwise` if input/output shapes match? Need to find examples.
- Horizontal/Vertical prims for `CombineByKindFusionRule` to finally cover those uses. Check we subsume
  `Combine`