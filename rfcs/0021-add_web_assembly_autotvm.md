- Feature Name: add web assembly autotvm
- Start Date: 2021-09-03
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

Add AutoTVM support to TVM's WASM backend, enhancing TVM's capability for web model deployment.

# Motivation
[motivation]: #motivation

The front-end web, as one of the most important components in the Internet ecology, has huge demands for 
AI deployment capability. Since front-end web involves a considerable number of different deployment 
scenarios such as pc, android, iOS, etc, and ML models has to be optimized for each of these scenes to reach 
the best performance, there is a pressing need to automatize the optimization process. However, currently TVM's WASM 
lacks AutoTVM support. A front-end web AutoTVM solution, as a result, is needed to make the 
optimization and deployment process more efficient.


# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

This section demonstrates steps to optimize models on web front-end with AutoTVM.

Step 1: Compile TVM as usual, then configure the python path.

Step 2: Ensure that the web environment has been configured according to Web/README.md.

Step 3: Start a `Tracker`:

``` python
  python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
```
Step 4: Start a `rpc_proxy`:

``` python
  python -m tvm.exec.rpc_proxy --example-rpc=1 --tracker=0.0.0.0:9190
```
Step 5: Open 0.0.0.0:8888, and server will be connected automatically.

Step 6: Run the tuning script:
``` python
  python tutorials/autotvm/tune_relay_wasm.py
```

The following will appear:
```
Extract tasks...
Tuning...
[Task  1/12]  Current/Best:    4.77/   4.77 GFLOPS | Progress: (1/1) | 5.70 s Done.
[Task  2/12]  Current/Best:    3.54/   3.54 GFLOPS | Progress: (1/1) | 4.24 s Done.
[Task  3/12]  Current/Best:    4.33/   4.33 GFLOPS | Progress: (1/1) | 7.18 s Done.
[Task  4/12]  Current/Best:    5.30/   5.30 GFLOPS | Progress: (1/1) | 8.28 s Done.
[Task  5/12]  Current/Best:    2.94/   2.94 GFLOPS | Progress: (1/1) | 4.06 s Done.
[Task  6/12]  Current/Best:    4.18/   4.18 GFLOPS | Progress: (1/1) | 6.15 s Done.
[Task  7/12]  Current/Best:    4.34/   4.34 GFLOPS | Progress: (1/1) | 6.35 s Done.
[Task  8/12]  Current/Best:    3.72/   3.72 GFLOPS | Progress: (1/1) | 5.06 s Done.
[Task  9/12]  Current/Best:    4.69/   4.69 GFLOPS | Progress: (1/1) | 4.37 s Done.
[Task 10/12]  Current/Best:    4.18/   4.18 GFLOPS | Progress: (1/1) | 4.20 s Done.
[Task 11/12]  Current/Best:    4.67/   4.67 GFLOPS | Progress: (1/1) | 4.58 s Done.
[Task 12/12]  Current/Best:    3.35/   3.35 GFLOPS | Progress: (1/1) | 8.20 s Done.
tune tasks end...
Compile...
output: /workspace/tvm/tutorials/autotvm/turning_out/resnet-18.wasm

```
Step 7: The tuning output will be saved at `/workspace/tvm/tutorials/autotvm/turning_out/resnet-18.wasm`.


# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

This rfc enhances the capabilities of previous RPC and web proxy.

- Unlike PC, Android, or iOS, web frontend has no system file apis. Thus, it is hard to push binary files generated during tuning process to web frontend like to other ends. 
Therefore, the intermediate product is saved in a specific directory, which is pulled by web front end for performance measurement.
- Expand 'python/tvm/autotvm/measure/measure_methods.py' to support emcc building, which 
  uses "emcc.create_tvmjs_wasm" to build wasm binary.
- Move temp tuning wasm binary to web dist directory, which is pulled by the web side, then the speed of the wasm binary is measured.
- Revise the tvmjs.RPCServer, so that wasm binary is pulled before RPCServer.


# Drawbacks
[drawbacks]: #drawbacks

- Auto schedule will be supported in the future.
- Paralleled AutoTVM implementation is needed.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

- The AutoTVM support is added to TVM's WASM backend for the first time.
- The AutoTVM capability of the WASM will align with that of other ends.

# Prior art
[prior-art]: #prior-art
TVM's WASM has not support AutoTVM yet. The purpose is to make up for the lack of this compatibility.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

Paralleled AutoTVM needs to be implemented.

# Future possibilities
[future-possibilities]: #future-possibilities

Auto schedule will be supported in the future.

