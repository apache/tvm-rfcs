* Feature Name: Meta Schedule (AutoTensorIR)
* Start Date: 2021-05-28
* RFC PR: TBD (apache/tvm-rfcs#0000)
* GitHub Issue: TBD (apache/tvm-rfcs#0000)

## 1. Summary

This proposal introduces Meta Schedule: a probabilistic scheduling DSL on TIR that unifies the approaches of AutoTVM and Auto Scheduler (Ansor). Meta schedule provides a pragmatic way to define the space of automatic tuning, extensibility in terms of all possible TIR schedule primitives like tensorization and loop partitioning, and customizability on every layer of the automation system.

Meta Schedule is our 3rd generation automatic scheduling system.

## 2. Motivation

**Scheduling and Design Space**

In TVM TensorIR, optimization of a TensorIR program is done via a sequence of transformations. For example, we reorder loops for better locality and we tensorize for specific hardware intrinsics. The process of invoking such a set of pre-defined transformations is called “**scheduling**”, and each transformation is called a “**schedule primitive**”. These primitives form a domain-specific language (DSL) describing the transformation of TensorIR programs. **Design space** is the set of all possible schedulings with respect to a TensorIR program.

**Problems with the Current Scheduling System**

* **Manual schedule**: Developers optimize their programs by manually invoking schedule primitives, i.e. explore points in the design space with humans in the loop. This can be a tedious and error-prone approach, hence the creation of AutoTVM and AutoScheduler (Ansor).
* **AutoTVM**: The automation system requires users to define “schedule templates” as the design space for each operator. Therefore, it is inextensible to hundreds of operators.
* **AutoScheduler (Ansor)**: It automatically generates schedule templates as the design space, according to a set of predefined “search rules”. However, it is non-trivial to extend AutoScheduler to new schedule primitives (tensorize, loop partition, software pipelining).
* The three systems above have isolated sets of APIs with several layers of their own abstraction, which are not only hard to learn, but also engineering-intensive to customize.

**Benefit of Meta Schedule**

* Succinct syntax, consistent APIs to TensorIR schedule with no other layer of abstraction.
* Provides unified APIs for implementing manual schedule, AutoTVM and AutoScheduler (Ansor).
* Extensibility to all the schedule primitives, including tensorization and loop partitioning. Almost no extra effort is needed to use a new primitive in auto-tuning.
* The automation infrastructure is customizable across every layer.

## 3. Guide-level explanation

In this section, we describe the syntax of meta schedule DSL, and how it could be used to describe and auto-generate the design space.

### 3.1. Manual Schedule

Meta schedule APIs are almost the same as TE or TensorIR scheduling. Here is an example of a manual schedule for matrix multiplication:

```python
# Designate a set of tile sizes
i_tiles = [16, 8, 8, 8]
j_tiles = [16, 8, 8, 8]
k_tiles = [256, 8]

# Tile the loops according to the tile sizes
i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
k_0, k_1           = sch.split(loop=k, factors=k_tiles)

# Organize the loops into “SSRSRS” 6-level tiles
sch.reorder(
    i_0, j_0, # S
    i_1, j_1, # S
    k_0,      # R
    i_2, j_2, # S
    k_1,      # R
    i_3, j_3, # S
)
```

In this example, the developers may tweak the tile sizes and measure the performance of the generated kernels to explore the opportunities of potential optimization.

Generally speaking, while writing a schedule, there are often some parameters that are hard to determine ahead of time, for example, tile sizes, unroll steps, or which tensor intrinsics to use. Developers may manually enumerate possible combinations of these unknown factors, and then pick the best schedule according to measurement results on their device.

### 3.2. AutoTVM-style Design Space Description

Meta schedule extends the schedule DSL with sampling instructions. When included in a schedule, these instructions parametrize the schedule from a single deterministic point to a space supported by random variables (tile size, etc.), making it possible for developers to describe the design space with meta schedule APIs.

We can extend the matmul example above to cover all possible tilings using these sampling instructions:

```python
# Sample tile sizes
i_tiles = sch.sample_perfect_tile(i, n=4)
j_tiles = sch.sample_perfect_tile(j, n=4)
k_tiles = sch.sample_perfect_tile(k, n=2)
# Tile the loops according to the random variables
i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
k_0, k_1           = sch.split(loop=k, factors=k_tiles)
# Organize the loops into “SSRSRS” 6-level tiles
sch.reorder(
    i_0, j_0, # S
    i_1, j_1, # S
    k_0,      # R
    i_2, j_2, # S
    k_1,      # R
    i_3, j_3, # S
)
```

### 3.3. Composite Schedule

Each schedule primitive handles only a very basic operation to transform the IR, for example, `split` only splits a loop into two. In the real world, the over-fine granularity of those primitives usually leads to repetitive and verbose scheduling code, as [mentioned](https://discuss.tvm.apache.org/t/rfc-tensorir-a-schedulable-ir-for-tvm/7872/43?u=junrushao1994) by developers in our community.

To counter this challenge, we allow users to register “composite schedules” that analyze the IR, and apply a set of schedule primitives correspondingly. For instance, a composite schedule may inspect a TensorIR block and decide whether we should call `compute_inline` on it. The composite schedule may use sampling instructions to fill in undecided choices.

Our system also ships with some built-in composite schedules, including:

* Multi-level tiling
* Inline pure spatial blocks
* Parallelize & vectorize & unroll
* Auto tensorize
* …

### 3.4. AutoScheduler-style Design Space Generation

AutoScheduler (Ansor) generates schedule templates by applying their SearchRules to each stage. Meta schedule treats a search rule as a composite schedule, and applies each composite schedule to each block of TensorIR to generate the design space.

### 3.5. Unifying manual schedule / AutoTVM / Ansor

In this section, we show that the design space induced by TE manual schedule, AutoTVM and Ansor are all subsets of meta schedule, and meta schedule further allows mixing those three styles to search jointly.

**Manual schedule**. The TE schedule is a special case of a meta schedule program, where there is no randomness introduced by sampling instructions. It is a single point in terms of design space.

**AutoTVM (Template-based tuning)**. Writing one or more schedule functions in meta schedule, potentially with sampling instructions, is a natural representation of AutoTVM’s schedule templates (knobs). The PPL generates one or more traces as the design space to explore.

**AutoScheduler (Ansor, Template-free tuning)**. As mentioned in the previous section, application of composite schedule rules generates the design space, which is equivalent to Ansor’s sketch generation.

**Mixing styles in design space definition**. By taking union of the spaces induced by the three special cases, our system allows developers to combine generic rules that Ansor provides and operator-specific scheduling.

## 4. Reference-level explanation

In this section, we introduce the underlying techniques for the automation system to extract and explore the design space.

### 4.1. Execution trace as the design space

**Trace**. To represent the design space defined by the meta schedule DSL, the underlying system records all the instructions users applied to the schedule class, including sampling and schedule primitives. We call this list of instructions a trace.

Executing the example above results in the following trace:

```
Instruction 0. Sample-Perfect-Tile
Instruction 1. Sample-Perfect-Tile
Instruction 2. Sample-Perfect-Tile
Instruction 3. Split
Instruction 4. Split
Instruction 5. Split
Instruction 6. Reorder
```

The trace is not directly user-facing, but a data structure inside the user-facing `Schedule` class that records the execution. The automation system extracts the trace and finds out the design space according to the sampling instructions.

**Union of traces**. Often a single trace is unable to represent the entire space. Therefore, more precisely, our system works on a list of traces as the union of potential design space.

**Fork a trace**. When two different decisions in the scheduling process are equally important to generate high-performance schedules, we allow forking the trace into two, and the design space is the union of the forked traces.

### 4.2. Exploring the Search Space

Meta Schedule provides several built-in exploration strategies to exhaustively or efficiently search for efficient schedules.

**Program replay**. A simple strategy that replays the schedule program that generates the PPL, and doesn’t use any advantage provided by the PPL.

**Random search**. Extracts the PPL, and repetitively re-executes the PPL by flipping coins purely randomly.

**Cost-model-guided evolutionary search**. A more efficient exploration strategy. We define two sets of rules:

* Mutator: defines how to jump to a point’s “neighbor” in the design space
* Postprocessor: sometimes it is non-trivial to statically determine the PPL, for example:
  * There is a hard requirement in CUDA that the maximum number of threads should not exceed 1024, but it is a random variable that cannot be determined before actually executing the PPL. In this case, we write a postprocessor that errors out when the condition is not satisfied.
  * The number of outer loops to be fused together depends on their extents, which are random variables. In this case, we annotate the maximum extent allowed on the block, and do actual fusion in a postprocessor.

Our evolutionary search algorithm uses mutators to find possible schedules in the design space, then applies postprocessors, and asks the cost model to predict its performance. After several iterations, the new schedules with the highest scores are finally compiled and measured on device. Epsilon-greedy is used in this process to balance exploitation and exploration.

### 4.3. Python first for flexibility & customizability

We engineer the system in a way that all levels are decoupled and open to customization, aiming at providing a playground for developers to try out new ideas and potentially deliver performance quickly.

While all the important APIs are implemented in C++ for efficiency, every part of the system can be easily switched to customized python implementation. For example,

**Customize design space in python**. Can be a python function that does the schedule

```python
def schedule_matmul(sch) -> sch:
    i, j, k = sch.get_loops(sch.get_block(“matmul”))
    i_tiles = sch.sample_perfect_tile(i, n=4)
    j_tiles = sch.sample_perfect_tile(j, n=4)
    k_tiles = sch.sample_perfect_tile(k, n=2)
    # Tile the loops according to the random variables
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
    k_0, k_1 = sch.split(loop=k, factors=k_tiles)
    # Organize the loops into “SSRSRS” 6-level tiles
    sch.reorder(
        i_0, j_0, # S
        i_1, j_1, # S
        k_0,      # R
        i_2, j_2, # S
        k_1,      # R
        i_3, j_3, # S
    )
    return sch
```

**Customize composite schedule in python**. We provide two ways to define a composite schedule in python:

Method 1. A simple decorator that converts a python function to a composite schedule

```python
@tir.as_composite_schedule(name="multi-level-tiling")
def multi_level_tiling(sch: Schedule, block: BlockRV) -> Union[Schedule, List[Schedule]]:
    ...
```

Method 2. Derive from `PyCompositeSchedule`, providing extra functionalities like initialization

```python
class MultiLevelTiling(PyCompositeSchedule):
    def initialize(...):
        ...
    def apply(...):
        ...
```

**Customize exploration strategies in python**. Developers can implement any search algorithm in python as well by deriving from `PySearchPolicy`.

**Other customizable components**. This list includes:

* Cost model
* Database
* Measure callbacks
* Feature extractor
* Program builder & runner
* Analysis methods
* ...

In a short summary, almost every component of the system is decoupled with each other and extensions could be easily plugged in.

### 4.4. Upstreaming Plan

[M3a] Core infrastructure of the PPL
* Instruction
* Trace
* Composite schedule
* Sampler
* Search policy
* Design space generator

[M3b] Host-side search infra
* Database
* Cost model
* Measure callback

[M3c] RPC-related search infra
* Measure input, build result, measure result
* Builder
* Runner

[M4a] Implementation of rules
* Various built-in composite schedules
* Various built-in mutators
* Various built-in postprocessors
* Automatic tensorization

[M4b] Relay integration

## 5. Drawbacks

We are not aware of any drawbacks of the proposed system.

## 6. Rationale and alternatives

The system is designed with the principle of minimalism: different from alternative solutions, we do not require any change in existing codebase, or extra APIs to learn. It could potentially lower the bar of using automation systems. 

Unifying manual scheduling, AutoTVM's semi automatic templates and AutoScheduler's (Ansor's) fully automatic sketch generation provides flexible way to balance injection new domain knowledge and automation.

Flexibility in customization allows quick try-out on new tasks, new strategies and new hardware targets without deep knowledge of the system.

## 7. Prior art

**Tensor Expression (TE)** in TVM is a DSL that decouples compute and schedule, which provides convenient ways to handcraft optimized kernels for different hardware targets.

**TensorIR** is the latest generation of TVM’s low-level IR. Its capability of eagerly applying schedule primitives opens the door for meta schedule, our proposed new-generation auto scheduling system.

**AutoTVM** is the 1st generation automation framework in TVM, which requires developers to implement per-operator scheduling templates, and the system could handle the tuning process.

**AutoScheduler (Ansor)** is the 2nd generation automation framework in TVM, whose built-in rules could automatically generate schedule templates for almost all the operators on CPU, GPU, etc.

## 8. Unresolved questions

**Supporting Control Flow and Assertions**

Right now the meta schedule DSL does not support control flow. Although we didn’t see any real-world use case right now, it is possible that it could appear in some future workloads.

A real-world issue we could see is that sampling may lead to wrong schedules on CUDA, e.g. the schedule results in a CUDA program that uses too much shared memory, too many threads, etc. In this case, we need to halt the program immediately. Therefore, introducing assertion may be helpful.

## 9. Future possibilities

**Unifying Manual Scheduling, AutoTVM and Ansor in TOPI**

Meta schedule provides an idiomatic approach to unify the three existing scheduling APIs in TVM:

* Manual schedules are meta schedules without sampling instructions
* AutoTVM templates are meta schedules where knobs are replaced by sampling instructions
* Each of Ansor’s search rules generates a snippet of a meta schedule

We further allow mixing different styles of scheduling and exploring the union space, which could help dispatch to different implementations.
