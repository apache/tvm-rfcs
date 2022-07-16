- Feature Name: Asynchronous stage in software pipeline
- Authors: [Masahiro Masuda](https://github.com/masahi), [Wuwei Lin](https://github.com/vinx13/)
- Start Date: (2022-06-17)

# Summary
This RFC proposes TIR constructs for invoking and synchronizing asynchronous operations, to express asynchrony **within the device code**.
Building on the propposed constructs, we introduce "asynchronous stage" in the TIR software pipeline.
Asynchrony is prevalent on the host (runtime) side, and this proposal is the first step toward bringing the notion of an asynchronous operation in the
generated code.

The most important component we should agree on is the model of synchronization: Coming up with a design that is general enough to be useful for diverse backends, while making sure that the chosen design can be translated to a low-level synchronization model of a particular backend, is highly non-trivial.
The approach described in this document is motivated by a use case for NVIDIA GPUs, but we took some cares so that the design can be adopted by other backends. For example, if a backend has an asynchronous DMA engine, vector and tensor unit, we can specify that each of them runs asynchronously in different stages in a pipeline, with necessary synchronization between them.

# Asynchronous stage in a software pipeline

### Background: What is a software pipeline, and what does the TIR software pipeline transform do?

Software pipeline is an optimization technique to improve instruction-level parallelism of a loop. For example, given this program:

```python
B = alloc([1])

for i in range(16):
    B[0] = A[i] + 1
    C[i] = B[0] + 1
```

the goal is to overlap the execution of two statements in the loop body, by letting the two statements operate on different iterations of the loop. This way, the second statement would no longer depend on the completion of the first statement in the same iteration.

The TIR software pipeline transform enables such transformation at the TIR level. We annotate the loop in the above program to specify, for each statement in the loop, the “stage” and the “order” in the pipeline:

```python
sch = ...
sch.annotate(i, "software_pipeline_stage", [0, 1])
sch.annotate(i, "software_pipeline_order", [0, 1])
```

Given the annotation above, the TIR software pipeline transform would break up the loop into three parts: prologue, pipeline body and epilogue. Different “stage” in the pipeline body become independent of each other, and the integer value of “stage” tells how many iterations each statement goes ahead of its consumer.

```python
B = alloc([2])

# Prologue
B[0] = A[0]

# Body
for i in range(15):
    B[(i + 1) % 2] = A[i] + 1
    C[i] = B[i % 2] + 1

# Epilogue
C[15] = B[1] + 1
```

The two statements in the body can potentially run in parallel, if the underlying HW supports out-of-order execution.

### Making parallelism more explicit: Asynchronous pipeline

What’s currently available today is, after all, a “software” pipeline: whether or not independent statements actually run in parallel is up to the underlying HW, and programmers have little control over it. Moreover, for in-order processors like Hexagon DSP, this transformation alone would not help.

The goal of this work is to support HW-backed asynchrony in the pipeline. Asynchronous data movement is becoming increasingly important in GPU computing, and NPUs typically have multiple kinds of asynchronous units (DMA copy, vector & matrix compute etc). To exploit such hardware features, it’s essential that we express all kinds of available asynchronies in the IR.

A user of the TIR software pipeline transform will be able to specify which pipeline stage should become asynchronous by an additional annotation. For example, given the  annotation specifying that the first stage in the pipeline be made async,

```python
for i in range(16):
    B[0] = A[i] + 1
    C[i] = B[0] + 1

...
sch.annotate(i, "software_pipeline_stage", [0, 1])
...

# "0" refers to the 0-th stage, corresponding to the first element in the list [0, 1].
sch.annotate(i, "software_pipeline_async_stages", [0])
```

we generate the following IR. An asynchronous block is decorated with the `async_scope` attribute, and further enclosed in the scope `async_commit_queue(0)`.  Synchronization is expressed by a scope `async_wait_queue(0, 1)`.

```python
B = alloc([2])

# Prologue
async_commit_queue(0):
   async_scope:
      B[0] = A[0]

# Body
for i in range(15):
    async_commit_queue(0):
       async_scope:
          B[(i + 1) % 2] = A[i] + 1

    async_wait_queue(0, 1):
       C[i] = B[i % 2] + 1

# Epilogue
async_wait_queue(0, 0):
   C[15] = B[1] + 1
```

The proposed async constructs are intentionally more general / abstract than what's needed by the TIR software pipeline, in the hope that
they would find their uses in more general settings. In particular, synchronization is done in terms of "queue": It is an abstract entity
associated with each asynchronous unit, and it tracks invocations and completions of asynchronous operations in the FIFO order.

**Semantics of the proposed scope annotations**
- `async_commit_queue(i)`: Group one or more invocations of async operations in the given scope, and “commit”(or push) them to the queue `i`. The exact interpretation of “committing” can be up to each backend, but informally it signifies that a group of async operations are now in-flight. A group of operations committed together is awaited as one chunk. Groups committed to the same queue complete in the FIFO order.

- `async_wait_queue(i, N)`: Block until only `N` **most recent** committed groups are still in-flight in the queue `i` . In other words, if there are `M` committed groups in-flight in the queue `i`, when reaching the synchornization scope `async_wait_queue(i, N)`, `M - N` oldest committed groups would be forced to complete. `N` doesn’t have to be a constant, but some backends may require a constant count (e.g. PTX)

They are inspired by the corresponding async data movement instructions in CUDA (PTX): [`cp.async.commit_group`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group) and [`cp.async.wait_group`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group).
The CUDA counterparts do not have the notion of “queue”, since there is only one kind of async operation (copy from global to shared memory) supported by the current generation of NVIDIA GPU (Ampere, at the time of writing). So "commit" and "wait" always refer to the same internal queue.

To support more general cases where there could be multiple kinds of asynchronous units, each of which has its own queue(s), `async_commit_queue` and `async_wait_queue` take the “queue”parameter. It can be an arbitrary integer, as long as it is used consistently. Moreover, it does not have to be a constant. However, in the current usage by the TIR software pipeline, "queue" coincides with the notion of "stage", and thus it is always an integer constant.

**The role of async_scope**. `async_scope` is represented by `AttrStmt` with key `tir::attr::async_scope`. It is inserted to let later transform passes know that the enclosed statement is intended to run asynchronously. This way, the actual lowering to target-dependent asynchronous instructions
can happen much later in the compilation flow, rather than before the software pipeline transform using tensorization. For example, rewriting of global to shared memory copy by CUDA-specific `cp.async` can be made simpler if the rewrite happens after buffer flattening and loop vectorization passes.

### `wait(in-flight-count)` vs `wait(finished-count)`

 It would be more intuitive if the semantics of `wait(N)` was “Wait until the oldest N async operations have completed”. But that would make translation to the corresponding PTX instruction more complicated, since we additionally need to keep track of the “number of async operations in-flight” at each synchronization point, and make that an additional argument to `async_wait_queue` so that we can do subtraction during translation of `async_wait_queue` to `cp.async`.

One of the pros of `wait(in-flight-count)` semantics is that, it is trivial to let all in-flight async operations to complete: `wait(0)`. The alternative semantics would require, again, precise tracking of the number of async operations in-flight at the desired sync point.


### More examples

**Three stages of compute, where the first two stages are async**. The second stage is both an async producer and consumer. This example demonstrates the use of the “queue” parameter. Note that there is no distinction of asynchronous copy or compute.

```python
B = alloc([1])
C = alloc([1])

for i in range(16):
    B[0] = A[i] + 1
    C[0] = B[0] + 1
    D[i] = C[0] + 1
```

```python
sch = ...
sch.annotate(i, "software_pipeline_stage", [0, 1, 2])
sch.annotate(i, "software_pipeline_order", [0, 1, 2])
# The first and second statements are async, and they are in different stages
sch.annotate(i, "software_pipeline_async_stages", [0, 1])
```

```python
B = alloc([2])
C = alloc([2])

# Prologue
for i in range(2):
   async_commit_queue(0):
      async_scope:
         B[i % 2]  = A[i] + 1

   if 1 <= i:
      async_commit_queue(1):
         async_wait_queue(0, 1):
            async_scope:
               C[(i - 1) % 2] = B[(i - 1) % 2] + 1

# Body
for i in range(14):
   # Stage 0
   async_commit_queue(0):
      async_scope:
         B[(i + 2) % 2]  = A[i + 2] + 1

   # Stage 1
   async_commit_queue(1):
      async_wait_queue(0, 1):
         async_scope:
            C[(i + 1) % 2] = B[(i + 1) % 2] + 1

   # Stage 2
   async_wait_queue(1, 1):
      D[i] = C[i % 2] + 1


# Epilogue
for i in range(2):
   if i < 1:
     async_commit_queue(1):
        async_wait_queue(0, 0):
           async_scope:
              C[(i + 15) % 2] = B[(i + 15) % 2] + 1

   if i < 1:
      async_wait_group(1, 1):
         D[(i + 14) % 2] = C[(i + 14) % 2] + 1
   else:
      async_wait_group(1, 0):
         D[(i + 14) % 2] = C[(i + 14) % 2] + 1

```

**Multi-stage pipelined GEMM where the shared memory copy is 4x multi-buffered + async, and shared to local copy is double-buffered**. This example uses a highly non-obvious annotation below and exercises the nested pipelining feature in the TIR software pipeline transform.

```python
sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, 2, 3, 3])
sch.annotate(k0, ann_key="software_pipeline_order", ann_val=[0, 1, 3, 2, 4])
sch.annotate(k0, ann_key="software_pipeline_async_stages", ann_val=[0])

sch.annotate(k1, ann_key="software_pipeline_stage", ann_val=[0, 0, 1])
sch.annotate(k1, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
```

`async_commit_queue` encloses both copies to `A_shared` and `B_shared`, so that the two copies can be awaited as one chunk.

```python

# Prologue
A_local = [2, ...]
B_local = [2, ...]
A_shared = [4, ...]
B_shared = [4, ...]

for i in range(3):
   async_commit_queue(0):
      async_scope:
         A_shared[i] <- global[...]

      async_scope:
         B_shared[i] <- global[...]

   if 2 <= i:
      async_wait_queue(0, 2):
         A_local[0] <- A_shared[0, ...]
         B_local[0] <- B_shared[0, ...]

# Body
for i in range(125):
   async_commit_queue(0):
      async_scope:
         A_shared[(i + 3) % 4] <- global[...]

      async_scope:
         B_shared[(i + 3) % 4] <- global[...]

   async_wait_queue(0, 2):

      A_local[1] <- A_shared[i % 4, ...]
      B_local[1] <- B_shared[i % 4, ...]

      compute(A_local[0], B_local[0])

      A_local[0] <- A_shared[(i + 1) % 4, ...]
      B_local[0] <- B_shared[(i + 1) % 4, ...]

      compute(A_local[1], B_local[1])

# Epilogue
for i in range(3):
   async_wait_queue(0, 1 - i):

      A_local[1] <- A_shared[0, ...]
      B_local[1] <- B_shared[0, ...]

      compute(A_local[0], B_local[0])

      if i < 2:
          A_local[0] <- A_shared[0, ...]
          B_local[0] <- B_shared[0, ...]

      compute(A_local[1], B_local[1])
```

### Implicit vs explicit approach to synchronization

The model of async synchronization adopted by CUDA can be categorized as an “implicit” one: Instead of saying “Wait for this operation to complete”, it says “Wait until only N most recent async operations are in flight”, or equivalently, “Wait until M oldest async operates have completed”, where M = “number of async operations in flight” - N.

In contrast, a standard and intuitive approach is more explicit, e.g. wait for the operation associated with “this” token / future to complete etc. This is true for “async-await” in general-purpose languages, [Async](https://mlir.llvm.org/docs/Dialects/AsyncDialect/) and [NVGPU](https://mlir.llvm.org/docs/Dialects/NVGPU/) dialects in MLIR, for example.

In general, the explicit approach is probably more preferable, since

- It makes it obvious which operation is waiting on which
- It is less stateful (less assumption on how the underlying HW should work)
- It naturally handles synchronization in the presence of control flow (since we can only wait on an operation that has actually happened).

These properties may help if we want do some analysis of async programs.

The current design started from and has stayed with CUDA’s implicit synchronization model based on counting, primarily because it makes mapping to the corresponding PTX instructions trivial. We can adopt the explicit model instead, if we have a good way to translate token-based synchronization to the counting one for PTX. So far, we do not have a good solution for this. MLIR has adopted the token abstraction, but they have not solve this problem either: Their `DeviceAsyncWaitOp` has [an optional attribute `numGroups`](https://mlir.llvm.org/docs/Dialects/NVGPU/#nvgpudevice_async_wait-mlirnvgpudeviceasyncwaitop) that directly corresponds to "in-flight count", and they basically generate either `wait(numGroups)` or `wait(0)`, [in their translation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp#L426-L427) of  `DeviceAsyncWaitOp` (token based) to PTX `cp.async` (counting based). `wait(0)` is always correct but least precise / efficient.

(The following is highly speculative) On the other hand, translation from “count” to “token” seems more feasible: At each synchronization point, a backend presumably maintains the number and the order of pending async operations. Given the count, it should be possible to derive the correct token from the corresponding ordered list of tokens.

It’s also worth noting some cons of the token-based synchronization, in the context of the TIR software pipeline. First, it is not obvious how a token should be represented at all. It would probably be an integer, but each backend would probably have its own different representation. Second, expressing dependencies via tokens would be natural if what we generate is a DAG-like structure. But the output of the TIR software pipeline transform is still a loop, without unrolling: We would need to be able to refer to “the token associated with an async operation from three iterations ago”, for example, but that is a bit awkward to express. We would end up maintaining a circular buffer of tokens, in addition to the circular buffer of multi-versioned buffer copies.

### Where to put `async_commit_queue`?
Although `async_commit_queue` can be attached to each async operation individually, we group multiple async invocations into one `async_commit_queue` if there are multiple async operations in the same stage.

For example, given the annotation,

```python
sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, 3])
sch.annotate(k0, ann_key="software_pipeline_async_stages", ann_val=[0])
```

`async_commit_queue(0)` would enclose the first two blocks.

However, if the order is given by

```python
sch.annotate(loop, ann_key="software_pipeline_order", ann_val=[0, 2, 1])
```

, the two async blocks are interleaved with their consumer block in the middle. In such cases, we need to attach `async_commit_queue` for each async block. An example transformation is shown toward the bottom of this document.

### Where to put `async_wait_queue` ?

We must put wait before the consumer of async ops, so for example the following is correct:

```python
for i in range(3):
   async_commit_queue(0):
      async_scope:
        A_shared[i] <- global[...]
        ...

   if i <= 2:
      async_wait_queue(0, 2):
         A_local[0] <- A_shared[0, ...]


for i in range(125):
   async_commit_queue(0):
      async_scope:
        A_shared[(i + 3) % 4] <- global[...]
        ...

   async_wait_queue(0, 3):
      A_local[1] <- A_shared[i % 4, ...]
      ...

   async_wait_queue(0, 2):
      A_local[0] <- A_shared[(i + 1) % 4, ...]
      ...
```

But the second wait subsumes the first one (since it allows less in-flight ops), so we may as well generate:

```python
for i in range(125):
   async_commit_queue(0):
      async_scope:
        A_shared[(i + 3) % 4] <- global[...]
        ...

   async_wait_queue(0, 2):
      A_local[1] <- A_shared[i % 4, ...]
      ...

      A_local[0] <- A_shared[(i + 1) % 4, ...]
      ...
```

(Actually, the first wait `async_wait_queue(0, 3)` is redundant, since `async_wait_queue(0, 2)` in the previous iteration makes sure that there are only two in-flight ops, and we only issue one more async op in the current iteration before `async_wait_queue(0, 3)`, i.e. the number of in-flight ops is already three.)

### How to determine the “in-flight count” for each `async_wait_queue` ?

Let's define two monotonically increasing indices. They are imaginary, in the sense that the actual indices they write to / read from have `mod N`  applied to them, where `N` is the multiplicity of multi buffering (`max_stage` + 1, to be exact).

- The producer head: The index the latest async operation has written into.
- The consumer head: The index a consumer of async results reads from at the current iteration.

TIR software pipeline transform makes these indices explicit and readily available.

**A simple observation**: The correct in-flight count is exactly `producer_head` - `consumer_head`.

For example, below the async producer writes to `i + 3`, and two consumers read from `i` and `i + 1`. The corresponding in-flight counts are `(i + 3) - i` and `(i + 3) - (i + 1)`.

```python
for i in range(125):
   async_commit_queue(0):
      async_scope:
         A_shared[(i + 3) % 4] <- global[...]
         ...

   async_wait_queue(0, 3):
      A_local[1] <- A_shared[i % 4, ...]
      ...

   async_wait_queue(0, 2):
      A_local[0] <- A_shared[(i + 1) % 4, ...]
```

The above access pattern is ideal in terms of the order of accesses to the async result. It is taken from a complicated nested software pipeline example, using the annotation:

```python
sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, 2, 3, 3])
sch.annotate(k0, ann_key="software_pipeline_async_stages", ann_val=[0])
```

But if a user mistakenly uses a slightly different annotation below, the third block, which is a consumer of the async ops in the first and second block, is put into the same stage as the async producers:

```python
sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, 0, 3, 3])
```

In such cases, we need to force all pending asyncs ops to complete before the async result is accessed by the consumer in the same stage. The result of transformation looks like this:

```python
for i in range(125):
   async_commit_queue(0):
      async_scope:
         A_shared[(i + 3) % 4] <- global[...]
         ...

   async_wait_queue(0, 0):
      A_local[1] <- A_shared[i % 4, ...]
      ...

      A_local[0] <- A_shared[(i + 3) % 4, ...]
      ...
```

Note that the index `i + 3` is both produced and consumed in the same iteration. So before we access `A_shared[(i + 3) % 4, ...]`, we need to put `async_wait_queue(0, 0)`. `(i + 3) - (i + 3) = 0`, so the math checks out here too.

**Waiting across pipeline body and epilogue boundary**. In this example, there is no async producer in the epilogue. Since the prologue and body have issued 128 async ops in total, the producer head can be determined as 127. Two consumer access copies at the indices `i + 125` and `i + 126`.

```python
for i in range(3):
   async_commit_queue(0):
      async_scope:
         shared[i] = ...

   ...

for i in range(125):
   async_commit_queue(0):
      async_scope:
         shared[(i + 3) % 4] = ...

   ...

# Two consumers but no async producer in the epilogue
for i in range(3):
   # in_flight_count = 127 - (i + 125)
   async_wait_queue(0, 2 - i):
      local[...] = shared[(i + 125) % 4]
      ...

   if i < 2:
     # in_flight_count = 127 - (i + 126)
     async_wait_queue(0, 1 - i):
        local[...] = shared[(i + 126) % 4]
        ...
```

If either of async operations in the prologue or body are predicated with a non-trivial condition, we cannot tell how many async ops have been actually issued. In this case, we put `async_wait_queue(0, 0)` in the epilogue (flush all pending async ops) and don’t try to do something clever.

**A producer with non-trivial predicate in the epilogue**. There is also a case where there are both an async producer and its consumer in the epilogue, but the producer is predicated so that the number of counts from the expression `producer_head` - `consumer_head` is not valid for all iterations. In such cases, we can generate two different in-flight counts enclosed in if-then-else, one of the counts being 0 (”give up”). See [this](https://github.com/masahi/tvm/blob/42edd5c74920846ce0805ee75537d5b392ce64dc/tests/python/unittest/test_tir_transform_inject_software_pipeline.py#L1223-L1226) test case for more details.

**Interleaved async producers, a consumer in between**. Given a program

```python
A_shared = alloc([1])
B_shared = alloc([1])

for i in range(16):
    A_shared[0] = A[i]
    B_shared[0] = B[i]
    compute(A_shared[0], B_shared[0])
```

and annotation which makes the two async blocks interleaved with its consumer, the second block:

```python
sch.annotate(loop, ann_key="software_pipeline_stage", ann_val=[0, 0, 3])
sch.annotate(loop, ann_key="software_pipeline_order", ann_val=[0, 2, 1])
sch.annotate(loop, ann_key="software_pipeline_async_stages", ann_val=[0])
```

the result of transformation looks as follows. Note that there are two `async_commit_queue` for the same stage.

```python
A_shared = alloc([4])
B_shared = alloc([4])

for i in range(3):
    async_commit_queue(0):
       async_scope:
          A_shared[i] = A[...]

       async_scope:
          B_shared[i] = B[...]

for i in range(13):
    async_commit_queue(0):
       async_scope:
          A_shared[(i + 3) % 4] = A[...]

    async_wait_queue(0, 5):
       compute(A_shared[i], B_shared[i])

    async_commit_queue(0):
       async_scope:
          B_shared[(i + 3) % 4] = B[...]

for i in range(3):
   ...
```

The in-flight count at `compute` is 5: From `A_shared`'s perspective, it allows for `(i + 3) - i` async operations to be in flight, while from `B_shared`'s perspective, the producer head at `compute`  points to the copy done by the previous iteration, so the in-flight count is calculated as `((i - 1) + 3) - i`. The sum of the two in-flight counts gives 5.

It’s not hard to see that the count 5 is the right one. The five most recent async groups that are allowed to be in flight at `compute`, at the iteration `i`, is:

- Copy to `A_shared` at the same iteration, `i`
- Copy to `B_shared` at the iteration `i - 1`
- Copy to `A_shared` at the iteration `i - 1`
- Copy to `B_shared` at the iteration `i - 2`
- Copy to `A_shared` at the iteration `i - 2`

Thus, both copies at the iteration `i - 3` will be forced to complete by `wait(5)` at the iteration `i`. And those copies are exactly what’s get consumed by `compute` at the iteration `i`.

### Assumption & limitation

- The relative order of an async producer and its consumer should be “reasonable”. Some effort has been done to support tricky and unusual cases, but it’s highly possible that some unexpected input programs and their annotations would cause invalid IR to be generated. Unresolved question: Should all pipeline configurations that pass the existing validity check be supported by async pipeline? Is there a way to derive the wait count statically that works for all possible cases?
- Control flow. If an async block is predicated so that we cannot tell exactly how many times it would be executed, we can only do `wait(0)`. It is unclear if such case would come up in practice. In the worst case, we can allocate a local variable that dynamically tracks the number of async operations that have actually happened. For now, this proposal will not implement such dynamic counting, but if it turns out that the current support is too limiting, we can revisit this approach. (Note for PTX: `cp.async(N)` apparently requires `N` to be an integer constant, so dynamic counting won’t work there.)

### Implementation status
The implementation, including test cases, is complete and ready to be upstreamed as soon as this RFC is accepted. The test cases include an end to end demonstration of pipelining transform and lowering to actual asynchronous instructions, runnable on an NVIDIA Ampere GPUs.
