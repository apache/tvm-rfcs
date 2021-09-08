- Feature Name: add web assembly autotvm
- Start Date: 2021-09-03
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

Add autotvm capability for web assembly, enhancing TVM's compatibility for web model deployment.

# Motivation
[motivation]: #motivation

Front-end web, as one of the most important scenarios in the Internet ecology, has huge demands for 
AI deployment capability. Since front-end web has quite a number of deployment scenarios like pc, android, 
iOS, applets, etc. To reach the best performance, there is a requirement to run an ML network in these scenarios.
However, currently TVM does not have web autotvm support, and the existing autotvm solution only supports autotvm 
on x86, arm, android, ios, etc. Therefore, We need a solution of autotvm in front-end web to be more 
efficient to deploy models for web.


# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

This section will demonstrate that how to use TVM to tune autotvm on the web front end, step by step.

Step 1: Compile TVM as usual then configure the python path.
Step 2: Ensure that the web environment has been configured according to Web/README.md.
Step 3: Start Tracker;

``` python
  python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
```
Step 4: Start rpc_proxy

``` python
  python -m tvm.exec.rpc_proxy --example-rpc=1 --tracker=0.0.0.0:9190
```
Step 5: Open 0.0.0.0:8888 ,then server will be auto connected.

Step 6: Run the tunning script
``` python
  python tutorials/autotvm/tune_relay_wasm.py
```

then will see:
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
Step 7: Got the tunning output on /workspace/tvm/tutorials/autotvm/turning_out/resnet-18.wasm


# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

This capability will expand the capabilities of previous RPC and web proxy.

- Due to there are no system file apis like PC, Android and IOS on the web frontend, it is hard to push binary files during tunning process like other end. Therefore, we put the intermediate product into a specific directory, which will be pulled and used to measure performance from web front-end.
- Expand 'python/tvm/autotvm/measure/measure_methods.py' to support emcc building, which 
  uses "emcc.create_tvmjs_wasm" to build wasm binary.
- Mv temp tunning wasm binary to web dist directory, which will be pulled from the web side, then the speed of the wasm binary will can be calculated.
- Change the tvmjs.RPCServer to pull wasm binary before RPCServer.


# Drawbacks
[drawbacks]: #drawbacks

- Auto schedule will be supported in the future.
- Paralleled AutoTVM need te be support.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

- The AutoTVM support of front-end web is not supported before.
- The Autotvm capability of the web will align with other ends.

# Prior art
[prior-art]: #prior-art
The previous implementation does not support AutoTunning in the web front-end scenario. The purpose of this time is to supplement this ability.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

Paralleled AutoTVM need te be support before it will be merged.

# Future possibilities
[future-possibilities]: #future-possibilities

Auto schedule will be supported in the future.

