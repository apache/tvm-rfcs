- Feature Name: DietCode: An Auto-Scheduler for Dynamic Tensor Programs
- Start Date: (2022-05-10)
- RFC PR: [apache/tvm-rfcs#xx](https://github.com/apache/tvm-rfcs/pull/xx)
- GitHub Issue: [apache/tvm#yy](https://github.com/apache/tvm/pull/yy)

# Summary
[summary]: #summary

We propose to integrate DietCode, an auto-scheduler for dynamic tensor programs,
to AutoTIR. DietCode offers the following features:
- A shape-generic search space to cover possible shapes in dynamic shape
  workloads.
- A dynamic-shape aware cost model to judge the quality of schedule candidates.
- Enhancement to the TVM CUDA codegen for imperfect tiling.

DietCode has been published by MLSys 2022 so please see [the
paper](https://proceedings.mlsys.org/paper/2022/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html)
for more details and evaluations. Meanwhile, the latest DietCode codebase is also publicly
available [here](https://github.com/UofT-EcoSystem/DietCode).

# Motivation
[motivation]: #motivation

Achieving high performance for compute-intensive operators in machine learning
workloads is a crucial but challenging task. Many machine learning and system
practitioners rely on vendor libraries or auto-schedulers to do the job. While
the former requires significant engineering efforts, the latter in TVM only supports
static-shape workloads in existing works. It is difficult, if not impractical,
to apply the existing auto-scheduler directly to **dynamic-shape workloads**, as
this leads to extremely long tuning time.

We observe that the key challenge faced by existing auto-schedulers when
handling a dynamic-shape workload is that they cannot construct a conclusive search
space for all the possible shapes of the workload, because their search space is
shape-dependent. To address this, this RFC aims to add dynamic-shape supports to
AutoTIR by integrating DietCode framework, which constructs **a shape-generic
search space and cost model** to auto-schedule dynamic-shape workloads
efficiently.

Our evaluation shows that DietCode has the following key strengths when
auto-scheduling an entire model end-to-end: 

1. reduces the auto-scheduling time by up to 5.88x less than the current
   auto-scheduler on 8 uniformly sampled dynamic shapes, and
1. improves performance by up to 69.5% better than the auto-scheduler and 18.6%
   better than the vendor library. All these advantages make DietCode an
   efficient and practical solution for dynamic-shape workloads.


# Guide-Level Explanation
[guide-level-explanation]: #guide-level-explanation

The existing experiments are largely conducted with auto-scheduler. However,
having been syncing with the AutoTIR team for quarters, we plan to integrate
this RFC to MetaSchedule (AutoTIR), because it provides more systematic
interface and cleaner integration path with less hacks.

To provide an example of additional information users are required to feed the
system (see https://github.com/UofT-EcoSystem/DietCode/tree/MLSys2022_AE for a
PoC design):

```python
# A symbolic shape constraint
T = tir.ShapeVar('T’)
I = tir.ShapeVar('I')
H = tir.ShapeVar('H')
# The candidate values of `T`
T_vals = range(1, 128)
wkl_insts = []
for t in T_vals:
  wkl_insts.append((t, 768, 768))
  wkl_insts.append((t, 768, 3072))
  wkl_insts.append((t, 3072, 768))


task = Task(func=Dense,
            args=(16*T, I, H),
            shape_vars=(T, I, H),
            wkl_insts=wkl_insts
            wkl_inst_weights=([1. for _ in T_vals],))
```

To enable auto-scheduling for dynamic shape workloads, users only need to:
1. Have `ShapeVar` in the TE/TensorIR compututation.
2. Specify the weight/distribution of each shape value.

Notes:
1. Symbolic constraint is required additional in Relay, but could be inferred
   automatically after Relax is introduced;
2. The proposed interface does not change any existing functionality.

# Reference-Level Explanation
[reference-level-explanation]: #reference-level-explanation

Here is an overview of the DietCode framework design.

<img src="https://raw.githubusercontent.com/UofT-EcoSystem/DietCode/main/docs/figures/DietCode.jpg" width="61.8%" />

- We accept the shape variables and the workload instances from the programmer.
  In the case when they are not detected, the auto-scheduler treats the workload
  as static and applies and current workflow on it.
- We construct **a shape-generic search space that consists of micro-kernels**,
  an incomplete program that carries out a tile of the complete computation, to
  efficiently support dynamic-shape workloads. 
  
  We use the hardware constraints (e.g., the maximum number of threads, the
  amount of shared and local memory) rather than the shape information to
  determine the micro-kernel candidates. Those candidates serve as the building
  blocks and are executed repeatedly to carry out a workload instance (defined
  as an static-shape instance of the dynamic-shape workload).
- We build a **micro-kernel-based cost model**. The key insight is that the cost
  of a complete program *P* that is made up of a micro-kernel *M* can be
  decomposed into two parts: 
  
  1. A shape-generic cost function *f*<sub>MK</sub> that predicts the cost of
     *M*, and
  1. A shape-dependent adaption cost function *f*<sub>adapt</sub> that defines
     the penalty of porting *M* to *P*.
  
  While *f*<sub>MK</sub> is a function that has to be learned and updated by
  real hardware measurements during the auto-scheduling process,
  *f*<sub>adapt</sub> is a simple term that can be evaluated using the core
  occupancy and the padding ratio (in other words, it does not require feature
  extraction from the schedules).
- We generate one kernel per workload instance and use the scikit-learn
  framework to train a decision tree dispatcher to map the workload instance to
  its corresponding kernel. The decision tree will be output in predicate-only
  format for efficient runtime dispatching and embedded as part of the host
  code. As an example, one possible auto-scheduling outcome can look like the
  following:
  ```C++
  __global__ void default_function0(float* X, float* W, float* Y) {...}
  __global__ void default_function1(float* X, float* W, float* Y) {...}
  __global__ void default_function2(float* X, float* W, float* Y) {...}

  // host code
  if (T < 16)
    call(default_function0)
  else if (T < 64)
    call(default_function1)
  else
    call(default_function2)
  ```
  Because everything can be included in a single `PackedFunc` object, the
  workflow is fully compatible with the Relay workflow.

# Drawbacks
[drawbacks]: #drawbacks

- The current compilation workflow generates one program per input shape.
  Although we can merge those static-shape programs into a single dynamic-shape
  program like the following code snippet:
  ```CUDA
  __global__ void default_function(float* X, float* W, float* Y,
                                   const int T)
                                   // Note the `T` here.
  ```
  Our evaluations indicate that this program has at least 5% worse performance
  compared with the static-shape alternatives. Hence, we decide to sacrifice the
  binary size for the runtime performance, which can potentially be problematic
  when the hardware resources are limited.

# Rationale and Alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

There is an approach proposed by [Nimble](https://arxiv.org/pdf/2006.03031.pdf),
which partitions a range of dynamic shape to buckets and tunes one kernel for
each bucket. We could, of course, implement this approach to the current
auto-scheduler and AutoTIR. However, as evaluated in the DietCode paper, this
approach is not guaranteed to achieve better performance as static shapes.

# Prior State-of-the-Arts
[prior-sotas]: #prior-sotas

- **Reuse-based Tuner** 

  Selective Tuning ([Cody Yu.
  2019](https://github.com/apache/incubator-tvm/issues/4188)) and ETO ([Jingzhi
  Fang et al. VLDB 2021](http://www.vldb.org/pvldb/vol15/p183-chen.pdf)) group
  workloads into clusters based on a set of pre-defined rules (e.g., similarity
  ratio in Selective Tuning) and reuse the same schedule in a single cluster.

- **Dynamic Neural Networks**

  Dynamic batching is a common graph-level optimization adopted by frameworks
  such as DyNet ([Graham Neubig et al. 2017](http://arxiv.org/abs/1701.03980)),
  Cavs ([Shizhen Xu et al. USENIX ATC
  2018](https://www.usenix.org/conference/atc18/presentation/xu-shizen)),
  BatchMaker ([Pin Gao et al. EuroSys
  2018](https://doi.org/10.1145/3190508.3190541)), and TensorFlow Fold ([Moshe
  Looks et al. ICLR 2017](https://openreview.net/forum?id=ryrGawqex)) for cases
  when the batch size is dynamic. 
  
  Nimble ([Haichen Shen et al. MLSys
  2021](https://proceedings.mlsys.org/paper/2021/hash/4e732ced3463d06de0ca9a15b6153677-Abstract.html))
  and DISC ([Kai Zhu et al. EuroMLSys
  2021](https://dl.acm.org/doi/10.1145/3437984.3458838)) both design a compiler
  to represent and execute dynamic neural networks. 
  
  Cortex ([Pratik Fegade et al. MLSys
  2021](https://proceedings.mlsys.org/paper/2021/hash/182be0c5cdcd5072bb1864cdee4d3d6e-Abstract.html))
  is a compiler-based framework on recursive neural networks. 
  
  Those works focus on the graph-level optimizations and therefore are
  orthogonal to DietCode, which operates on each individual layer. In fact,
  those graph-level solutions can also leverage DietCode for efficient operator
  code generation.

# Unresolved Questions
[unresolved-questions]: #unresolved-questions

- The current design does not support arbitrary shape dimensions. For better
  auto-scheduling outcomes, we expect that shape dimensions have to be specified
  beforehand.
- The proposed approach mostly works on NVIDIA GPUs and has not been tested on
  other hardware platforms.

# Future Possibilities
[future-possibilities]: #future-possibilities

- Evaluate more operator use cases.
- CPU Support

# Upstream Milestones
[upstream-milestones]: #upstream-milestones

We propose the following milestones for upstreaming, where each bullet point
corresponds to a PR with unit tests of roughly several hundred lines.

- [ ] Code Generation Support
  - Local Padding
  - Loop Partitioning
- [ ] Meta-Scheduler
  - Frontend Interface
  - Sketch Generation
  - Random Annotations
  - Program Measurer
  - Micro-Kernel Cost Model
  - Evolutionary Search
- [ ] Decision-Tree Dispatching

When testing, we will be following the same testing procedure with the
meta-scheduler. We do not require any extra hardware platforms. Our plan is to
use a dynamic-shape workload (i.e., dense from BERT and conv2d from ResNet-50)
and compare its performance numbers with those delivered by the meta-scheduler
on static-shape workloads. The performance difference is expected to be smaller
than 5%.
