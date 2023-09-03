- Feature Name: AutoTVM tuning with Subgraph Granularity
- Start Date: 2022-3-17
- RFC PR: [apache/tvm-rfcs#0064](https://github.com/apache/tvm-rfcs/pull/0064)
- GitHub Issue: N/A

# Summary
[summary]: #summary

This RFC introduces why and how we tune with subgraph granularity.

# Motivation
[motivation]: #motivation

During performance optimization for platform Xavier which has a Volta GPU in it, we found that tuning by AutoTVM with subgraph as granularity could bring performance improvement.  Because the data type of the subgraph's output may be different from the data type of the subgraph's anchor operator's output, this may change the task from memory-bound to compute-bound or change in reverse.
Let's take the subgraph in the figure below as an example. If we tune with the single convolution the output data type is 'Int32' but if we tune with the subgraph the output data type is 'Int8'. The former's data size is four times the latter one. But in the actual inference, the data type is 'Int8' same as the latter one. So the best config searched by tuning with a single operator maybe not be the best for the subgraph.
![image](assets/9999/subgraph-example.png)
We also run an example to verify the theory above.

- We wrote a schedule marked as ```ScheduleA``` for the subgraph above by hardcode and the latency of subgraph inferencing is 104 microseconds. Then we tuned the subgraph with single op as granularity. In the tuning log we found  ```ScheduleA``` and the latency recorded in the measurement result is 329 microseconds.
- The best schedule from the tuned log in the step above is marked as ```ScheduleB``` and the latency recorded in the measurement result is 237 microseconds. The latency of subgraph inferencing with ```ScheduleB``` is 224 microseconds.

From the example above we can tell AutoTVM would not find ```ScheduleA```,  the obvious better schedule. This means the tuning result is distorted, the distortion would be more obvious if the shape of the output became bigger.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

We propose a method to tune with subgraph as granularity by AUTOTVM. As we know, tuning with a single operator we need function ```fcompute``` which computes the output tensors of the operator and function ```fschedule``` which schedules the computation. With these two functions, we can build a measurable program running on the device. So the key problem is assigning function ```fcompute``` and function ```fschedule``` for subgraph. In this PR, we use the fucntion  ```fschedule``` of the anchor operator as the subgraph's function ```fschedule```. As for function ```fcompute```, its purpose is getting output tensors of the subgraph. The function ```LowerToTECompute::Lower``` can get the output tensors of the subgraph, so we can use these tensors as output of subgraph's function ```fcompute```. And a GLOBAL_SCOPE.tune_subgraph option is introduced to control tuning with single operator or subgraph, default vaule is `False`.
The whole process can breakdown into two major phases.
1. Task extracting and tuning.
1.1 Select best implementation to compute output for anchor operator and record implementation name `best_impl_name`
1.2 Lower subgraph to `outputs` by `LowerToTECompute`
1.3 Create subgraph tuning task name `task_name` with subgraph name and `iotensors` extracted from `outputs`.
1.4 Add subgraph task with `task_name`, `iotensors` and `best_impl_name`.
1.5 Create `workload` for subgraph tuning task with `task_name` and `iotensors`.
1.6 Set function `fcompute`  for subgraph by returning `outputs` .
1.7 Set function `fschedule`  for subgraph by querying table with `best_impl_name`.
1.8 Tune.

1. Building.
2.1 Apply the best history.
2.2 Select best implementation to compute output for anchor operator and record implementation name `best_impl_name`
2.3 Lower subgraph to `outputs` by `LowerToTECompute`
2.4 Create subgraph tuning task name `task_name` with subgraph name and `iotensors` extracted from `outputs`.
2.5 Create `workload` for subgraph tuning task with `task_name` and `iotensors`.
2.6 Lower schedule with the best config queried by `workload`.
2.7 Codegen.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

In steps 1.1 and 2.2 mentioned in the previous section, returning the best implementation name is kind of tricky. We can get the `best_plevel_impl` in function `select_implementation`, but sometimes the actual implementation name is not `best_plevel_impl.name`. For example, implementation for `conv2d_nchw.x86` is added like this.

```python
@conv2d_NCHWc_strategy.register("cpu")
def conv2d_NCHWc_strategy_cpu(attrs, inputs, out_type, target):
    """conv2d_NCHWc x86 strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d(topi.x86.conv2d_nchw),
        wrap_topi_schedule(topi.x86.schedule_conv2d_nchw),
        name="conv2d_nchw.x86",
    )
    return strategy
```

But `topi.x86.conv2d_nchw` wrappers another implementation.

```python
def conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype):
    layout = "NCHW"
    packed_out = conv2d_NCHWc(data, kernel, strides, padding, dilation, layout, layout, out_dtype)
    return unpack_NCHWc_to_nchw(packed_out, out_dtype)
```

The actual implementation name is `conv2d_NCHWc.x86`. So in function `select_implementation` we fixed this problem by getting the actual implementation name from the workload which is created in function `register_topi_compute`.

```python
def select_implementation():
  # ignore some codes

  if GLOBAL_SCOPE.tune_subgraph:
    # In some cases, one strategy's compute may call another compute.
    # So the impl name need to match with actual compute.
    if workloads[best_plevel_impl]:
        workload = workloads[best_plevel_impl]
        if best_plevel_impl.name != "injective.cpu" and best_plevel_impl.name != workload[0]:
          best_plevel_impl.name = workload[0]
  # value changed in python side will not effect C++ side,
  # so here need to pass new name to C++
  return best_plevel_impl, outputs[best_plevel_impl], best_plevel_impl.name
```

Because the value changed on the python side will not affect the C++ side, we need to pass the new name to C++. This results in the function `select_implementation` number of returned values changing to three. This should be noticed.


# Drawbacks
[drawbacks]: #drawbacks

There are three cases in which tuning with subgraph may not get better performance.

Case 1: Anchor operator has more than one implementation.
We register the subgraph tuning task by `outputs` in step 1.4. No matter how many implementations the anchor operator has, step 1.1 will only pick the implementation with the highest level and return outputs computed by it. So the subgraph tuning tasks may not contain the potential best implementation.

Case 2: Anchor operator's function `fcompute` needs value from config such as code block below. In step 2.2, computing output will call function `_pack_data` and the `cfg` suppose to be the best config of the subgraph. But in step 2.2 we don't know which subgraph the anchor operator belongs to yet, so we cannot get the right config from the best history and fallback to the default one. This may bring great performance regression.

```python
def _pack_data(cfg, data, kernel):
    n, _, ih, iw = get_const_tuple(data.shape)
    oc, ic, kh, kw = get_const_tuple(kernel.shape)
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    ......
```

Case 3: During task extraction subgraph is lowered by VMCompile, and pass `AlterOpLayout` is disabled, see [code](https://github.com/apache/tvm/blob/main/python/tvm/autotvm/task/relay_integration.py#:~:text=with%20tvm.transform.PassContext(opt_level%3Dopt_level%2C%20disabled_pass%3D%7B%22AlterOpLayout%22%7D)%3A). So during building phash we need to disable pass `AlterOpLayout` too, otherwise the subgraphs generated in the task extracting phase may be different from those generated in the building phase.


# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

The core problem of tune subgraphs is getting output tensors of subgraphs. Employing function `LowerToTECompute::Lower` can achieve minimal change to the current framework.

# Prior art
[prior-art]: #prior-art

Our implementation is inspired by auto-schedule.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

See Section Drawbacks.

# Future possibilities
[future-possibilities]: #future-possibilities

Resolve the 3 drawbacks listed above.
