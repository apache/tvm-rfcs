<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->
- Feature Name: (fill me in with a unique identifier, `my_awesome_feature`)
- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: [apache/tvm-rfcs#0014](https://github.com/apache/tvm-rfcs/pull/0014)
- GitHub Issue: [apache/tvm#8596](https://github.com/apache/tvm/issues/8596)

## 1. Summary


This proposal introduces Pipeline Executor: A runtime executor that by scheduling
splitted subgraph of relay graph in pipeline to implement task level parallism to
reduce compute latency.

## 2. Motivation



Currently more and more edge device inference deployments happen on SOC devices.
Since SOC devices have heterogeneous chipset like GPU, FPGA, CPU, DSP, etc. To reach the best
performance, there is a requirement to run an ML network in these heterogeneous chipsets.
However, currently graph executor does not have parallelism logic, and the existing data parallelism
solution only supports parallel on homogeneous chipset(device). Then, the only way to do batch processing
on heterogeneous devices with TVM is to treat a whole ML network as a schedule unit and run it on
different heterogeneous devices, but that would cause latency issue (low speed chipset becomes the
latency bottleneck for single data processing).

Therefore, we need a runtime executor that can provide parallel scheduling functionality
with a finer-grained schedule unit like subgraph (a group of operator with dependency relation)
to be more efficient to use SOC heterogeneous hardware resource to achieve a better performance.


### Benefits of Pipeline Executor

There are three benefits for Pipeline Executor

Pipeline Executor provides:
* Compute a single network on multiple backends in parallel to improve performance.

* Use RPC to perform distributed computation cross multiple remote devices.

* User can use Pipeline Executor to integrate pre-compute processing and pos-processing with
  network compute together and compute in same executor.

## 3. Guide-level explanation
Pipeline Executor is a runtime executor which implements pipeline execution logic for multiple
subgraphs and relies on graph_executor for operator storage and execution.

This section introduce the use case for Pipeline Executor.

* 1. Manually constructing pipeline subgraph from a network compute graph.
* 2. Manually contstructin pipeline subgraph configuration for dependency and target device...
* 3. Use pipeline_executor to build pipeline module with the said subgraph and configuration.
* 4. Use pipeline_executor to load pipeline module to run network in pipeline parallism mode.

### 3.1. Manually constructing pipeline subgraph from a network compute graph.
pipeline subgraph is subset of network compute graph, there are dependency relation
between different pipeline subgraph, each pipeline subgraph running on different backend
, the purpose of split network into pipeline subgraph is to do network compute on different
compute unit and pipeline them to reduce compute latency, following is example for network
compute graph split.

```python
import tvm
from ...ir import IRModule
from ...relay import transform, build_module
def pipeline_graph(expr, indices):
    """Split Graph Into A Group Of Subgraph
    Parameters
    ----------
    expr : tvm.relay.Expr
    indices : Array[int]
    Returns
    -------
    ret : Array[tvm.relay.IRModule]
    """

    def run_opt_pass(expr, opt_pass):
        """Exectue a relay pass"""
        assert isinstance(opt_pass, tvm.transform.Pass)
        mod = tvm.IRModule.from_expr(expr)
        mod = tvm.relay.transform.InferType()(mod)
        mod = opt_pass(mod)
        entry = mod["main"]
        return entry if isinstance(expr, tvm.relay.Function) else entry.body

    def _operator_idx_inc(expr, operator_current_idx):
        """Increase operator index"""
        if not isinstance(expr, tvm.relay.expr.Constant):
            operator_current_idx = operator_current_idx + 1

        return operator_current_idx

    def merge_constant_expr(constant_expr, expr):
        # merge constant express with a express
        # Parameters
        # ----------
        # constant_expr:
        #     constant expression
        # expr:
        #     expression to merge with constant expression

        # If body not let, then reached end of the express
        if not isinstance(constant_expr.body, tvm.relay.expr.Let):
            return tvm.relay.expr.Let(constant_expr.var, constant_expr.value, expr)

        return tvm.relay.expr.Let(
            constant_expr.var, constant_expr.value, merge_constant_expr(constant_expr.body, expr)
        )

    def _recursion(anf, operator_indx, pipeline_mods, indices, constant_expr):
        # Enumrate all operator of compute graph then split the compute graph
        # into a group subgraph.
        # Parameters
        # ----------
        # anf:
        #     ANF format expression
        # operator_indx:
        #     current operator indice
        # pipeline_mods:
        #     the subgraph list get storage in this variable
        # indices:
        #     Array of indices use to define the subgraph scope
        # constant_expr:
        #     constant defined before current operator

        # Do the split work
        if isinstance(anf, tvm.relay.Function):
            return tvm.relay.Function(
                anf.params,
                _recursion(anf.body, operator_indx, pipeline_mods, indices, constant_expr),
                anf.ret_type,
                anf.type_params,
                anf.attrs,
            )
        if isinstance(anf, tvm.relay.expr.Let):
            value = anf.value
            operator_indx = _operator_idx_inc(value, operator_indx)

            # record constan expr to make sure all sugraph can find correct
            # constant.
            if isinstance(value, tvm.relay.expr.Constant):
                if not constant_expr:
                    constant_expr = tvm.relay.expr.Let(anf.var, value, anf.var)
                else:
                    constant_expr = tvm.relay.expr.Let(anf.var, value, constant_expr)

            if isinstance(value, tvm.relay.expr.Call):
                if isinstance(value.op, tvm.ir.Op):

                    # if have expr a(b(c(d(e)))) and indexes are [1,2,3]
                    # then would get separate modules for a(b),c,d(e).
                    # the split area is a(b)[0,1] c[2,2] d(e)[2,3]
                    if indices and operator_indx == indices[0]:
                        indices.pop(0)
                        ann = _recursion(
                            anf.body, operator_indx, pipeline_mods, indices, constant_expr
                        )

                        # when current subgraph use previous subgraph constant,
                        # such constant may become free varaible due to the constant
                        # not exist, merge the previous constant with current subgraph
                        # to avoid such issue.
                        if constant_expr:
                            ann = merge_constant_expr(constant_expr, ann)

                        ann = run_opt_pass(ann, transform.ToGraphNormalForm())
                        mod = tvm.IRModule.from_expr(ann)
                        pipeline_mods.insert(0, mod)
                        return tvm.relay.expr.Let(anf.var, value, anf.var)
            return tvm.relay.expr.Let(
                anf.var,
                value,
                _recursion(anf.body, operator_indx, pipeline_mods, indices, constant_expr),
            )
        else:
            return anf

    pipeline_mods = []

    # operator count start from 0, then initial value get set into -1
    operator_indx = -1
    constant_expr = None
    subgraph_indices = indices.copy()
    anf = run_opt_pass(expr, transform.ToANormalForm())
    anf = run_opt_pass(anf, transform.InferType())
    ann = _recursion(anf, operator_indx, pipeline_mods, subgraph_indices, constant_expr)
    ann = run_opt_pass(ann.body, transform.ToGraphNormalForm())
    mod = tvm.IRModule.from_expr(ann)
    pipeline_mods.insert(0, mod)
    return pipeline_mods

#...
mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=dshape)
split = [11, 22]
mods = pipeline_graph(mod["main"], split)
```

### 3.2. Manually contstructin pipeline subgraph configuration for dependency and target device...
There are dependency between pipeline subgraph, for example we have 3 pipeline subgraph named
s1, s2, and s3, s2 input is s1 output and s2 output is s3 input, we need to construct a configuation
file to descript such dependency relation, such configuratin also need to involved "target" and
"device" information following is a example.

```python
mconfig = {"target_host": None, "mod_name": "default", "build": None, "params": None}
    mconfig1 = mconfig.copy()
    mconfig1["target"] = "cuda"
    mconfig1["dev"] = tvm.gpu[0]
    # third output is final output, second output for mod3, first for mod2
    # input
    mconfig1["pipeline"] = {
        "mod_indx": 1,
        "output": [
            {"output_indx": 1, "dependent": [{"mod_indx": 2, "input_name": "data_0"}]},
            {"output_indx": 2, "dependent": [{"mod_indx": 3, "input_name": "data_0"}]},
            {"output_indx": 3, "dependent": [{"mod_indx": 0, "input_name": "1"}]},
        ],
    }
    mod_config[mods[0]] = mconfig1

    mconfig2 = mconfig.copy()
    mconfig2["target"] = "llvm"
    mconfig2["dev"] = tvm.cpu(0)
    mconfig2["pipeline"] = {
        "mod_indx": 2,
        "output": [
            {"output_indx": 1, "dependent": [{"mod_indx": 3, "input_name": "data_1"}]},
        ],
    }
    mod_config[mods[1]] = mconfig2

    mconfig3 = mconfig.copy()
    mconfig3["target"] = "llvm"
    mconfig3["dev"] = tvm.cpu(0)

    mconfig3["pipeline"] = {
        "mod_indx": 3,
        "output": [{"output_indx": 1, "dependent": [{"mod_indx": 0, "input_name": "2"}]}],
    }
    mod_config[mods[2]] = mconfig3
``` 

### 3.3. Use pipeline_executor to build pipeline module with the said subgraph and configuration.

Pipeline executor provide a build function to compile and save the compile output into disk,
following is a example

```python
    with relay.build_config(opt_level=3):
        pipeline_mods, string_config = pipeline_executor.build_pipeline(
            mod_config, "<path to storage the build output>"
        )

```

### 3.4. Use pipeline_executor to load pipeline module to run network in pipeline parallism mode.

Pipeline executor works asynchronously. Unlike the graph executor that launches a task by calling a blocking
`run` API, we can kick off a task by calling a non-blocking `set_input` API in pipeline executor:
    set_input--> run
    set_input--> run
    get_ouput
    set_input-->run
    get_output
    get_output

`get_output` can be called anytime, and it will return an empty array if no output is ready.

following is one example

```python
#...

def get_output(outputs, module):
  suc = False
  output = pipeline_module.get_output()
  if len(output):
    curOutputs = [output.asnumpy() for data in output]
    outputs.append(curOutputs)
    suc = True

  return suc
    

pipeline_outputs = []
datas = []
for i in range(len(mods) + 1):
  datas.append(np.full(dshape, 3 + i).astype("float32"))
pipeline_module = pipeline_executor.create(pipeline_mods, string_config)

for data in datas:
    get_output(pipeline_outputs, pipeline_module)
    pipeline_module.set_input("data_0", data)
    pipeline_module.set_input("data_1", data, mod_idx=2)
    pipeline_module.run()
    get_output(pipeline_outputs, pipeline_module)

left = len(datas) - len(pipeline_outputs)
while(left > 0):
    left = left - 1 if get_output(pipeline_outputs, pipeline_module) else left
```

## 4 Reference-level explanation
This section introduces the underlying techniques for the pipeline executor.
The figure below briefly illustrates the workflow of the system

Pipeline executor architecture
![meta-schedule-workflow](../resources/pipeline-executor-arch.png)

Manually construct the subgraph
![meta-schedule-workflow](../resources/pipeline-executor-subgraph-split.png)

How pipeline executor runtime work
![meta-schedule-workflow](../resources/pipeline-executor-runtime.png)

The pipeline executor schedule logic
![meta-schedule-workflow](../resources/pipeline-executor-schedule.png)

The network pipeline compute effect
![meta-schedule-workflow](../resources/pipeline-executor-pipeline.png)


## 5. Drawbacks


Pipeline executor currently needs manually subgraph splitting and configuration construction.
Further graph splitting feature would do automatically split.

## 6. Rationale and alternative


Compute graph can get split into subgraph and pipeline execution can implement parallism
when these subgraph have dependency relation.


## 7. Prior art


**Schedule Primtive like Vectorize etc** the schedule primtive implement data parallism
on same device.

## 8. Unresolved questions


Automatically split compute graph

## 9. Future possibilities


Use Autotune to get best graph split solution
