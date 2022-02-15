# Module-based Model Runtime Interface for AOT

- Feature Name: module_based_model_runtime_for_aot
- Start Date: 2021-09-17
- RFC PR: [apache/tvm-rfcs#0046](https://github.com/apache/tvm-rfcs/pull/0046)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# **Summary**

This RFC describes a [Module-based Model Runtime
interface](https://discuss.tvm.apache.org/t/discuss-module-based-model-runtime-interface/5025) for
the [Ahead-of-Time Executor](https://discuss.tvm.apache.org/t/implementing-aot-in-tvm/9206), thereby
enabling its use from the TVM C++ Runtime.

# **Motivation**

The microTVM project has made significant progress towards an Ahead-of-Time Executor for compiled
Relay models. At the time of writing, it's now possible to codegen a TIR function which executes
Relay models that have known shapes, don't have graph-level control flow, and execute only on the
CPU device. Right now, the C runtime is the only such runtime environment which can interact with
this generated code. However, significant interest exists in enabling the C++ runtime to use the
Ahead-of-Time executor.

# **Guide-level explanation**

Users select the AOT executor at compile time through the traditional GraphExecutor compilation flow
(e.g. `[tvm.relay.build](http://tvm.relay.build)`) by including `--executor=aot` in the Target
[1]. The return value of `tvm.relay.build` in this case is an `AotExecutorFactory` Module
object. Users instantiate the AOT executor via `AotExecutorFactory` as they do with `GraphExecutor`:

```bash
ir_mod = tvm.parser.fromtext("""\
      #[version = "0.0.5"]
      def @main(%a : Tensor[(1, 2), uint8], %b : Tensor[(1, 2), uint8]) {
          %0 = %a + %b;
          %0
      }"""
    )

with PassConfig(opt_level=3):
  factory : AotExecutorFactory = tvm.relay.build(
       ir_mod, "llvm -executor=aot", module_name="my_mod")

aot_executor : AotExecutor = factory["my_mod"](tvm.cpu(0))
```

`AotExecutor` supports the traditional Module-Based Model Runtime Interface and can be used as a
user normally would `GraphExecutor`:

```bash
aot_executor.set_input("a", tvm.nd.array(np.ndarray([1, 2], dtype="uint8")))
aot_executor.set_input("b", tvm.nd.array(np.ndarray([3, 5], dtype="uint8")))
aot_exec.run()
output = aot_exec.get_output(0)
assert output.asnumpy() == np.ndarray([5, 7], dtype="uint8")
```

[1] NOTE: The target string is not the final place this customization should be made. However, it's
been the place where we've been putting runtime-related stuff. A separate RFC will split the Target
string into Target options (which affect tuning) and runtime options.

# **Reference-level explanation**

Already committed to TVM is the AotExecutorCodegen. This module produces a TIR top-level function
which invokes the Relay operators (implemented in TIR) in a correct order. An example is given
below:

```bash
PrimFunc([input1, input2, output]) attrs={"global_symbol": "tvmgen_my_mod_run_model", "runner_function": (bool)1} {
  // attr [(nullptr)] device_id = 0
  // attr [(nullptr)] device_type = 1
  tir.tvm_call_packed("tvmgen_my_mod_fused_add", input1, input2, output)
}
```

The AotExecutor is a runtime wrapper component around this function that needs to accomplish the
following to meet Module-based Model Runtime Interface:

1. Allocate input and output tensors as defined in the `run_model` function using the correct Device
   API.
2. Provide a mapping from relay parameter name to positional argument.
3. Invoke the generated TIR function and provide profiling.

In the future, AOT will support heterogenous execution e.g. allocating tensors and driving inference
on `DLDevice` other than `kDLCPU`. Note that to align this code generator with the sensitive
environment present on a bare-metal microcontroller, the TIR top-level function intentionally
presumes that the input and output tensors already live on the `DLDevice`. This allows the user to
decide whether the AotExecutor generic runtime component will be used to fill input tensors or
whether they prefer to handle this in their application (or e.g. through background DMA).

### Compiler ↔ Runtime Metadata

In order to implement (1) and (2) above, additional metadata about the `run_model` function needs to
be communicated from Compiler to Runtime:

- The mapping between Relay parameter name and TIR argument position
- The number of inputs and outputs
- The type of each parameter
- Information sufficient to choose a Device API to allocate memory for that data.

At present, Metadata is passed from Compiler to Runtime in several different ways:

1. Constant DLTensor can be bundled with code and supplied to `runtime::Module` via
   `runtime::MetadataModule`
2. Many non-DSO-exportable backends (`cuda`, `hexagon`, `metal`, `opencl`, `sdaccel`, `rocm`,
   `vulkan`) have adopted the convention of including a
   [`runtime::FunctionInfo`](https://github.com/apache/tvm/blob/main/src/runtime/meta_data.h#L106)
   (NOTE: distinct from `tvm::relay::transform::FunctionInfo`) in their serialization:

    ```bash
    /*! \brief function information needed by device */
    struct FunctionInfo {
      std::string name;
      std::vector<DLDataType> arg_types;
      std::vector<std::string> launch_param_tags;
    }
    ```

3. AotExecutorCodegen and GraphExecutorCodegen have adopted the practice of producing the
   graph-level
   [`tvm::relay::backend::ExecutorCodegenMetadata`](https://github.com/apache/tvm/blob/c3ace209253507dcb109c12ab8b82575fc668862/src/relay/backend/utils.h#L89):

    ```bash
    /*!
     * \brief Structure that can be optionally used by the executor codegen
     */
    class MetadataNode : public Object {
     public:
      /*! \brief input information for the main function */
      Array<String> inputs;
      /*! \brief number of outputs of the main function */
      int num_outputs = 1;
      /*! \brief the executor to be used to run the model */
      String executor = kTvmExecutorGraph;

      String mod_name = "";
    }
    ```

4. The recent AOTExecutor implementation has created `tvm::relay::transform::FunctionInfo` which
   communicates statistics about memory usage and I/O operation for each TIR operator and aggregate
   statistics for the top-level AOT function:

    ```bash
    struct FunctionInfoNode : public Object {
      Map<Target, Integer> workspace_sizes;
      Map<Target, Integer> io_sizes;
      Map<Target, Integer> constant_sizes;
      Map<Target, tir::PrimFunc> tir_primfuncs;
      Map<Target, Function> relay_primfuncs;
    }
    ```


Some duplication of information is already present. Likely this is due in part to the existing
middle-end compiler design, in which a separate `IRModule` is produced for each backend. This means
that any metadata which requires whole-program analysis must be computed by an upstream TIR pass and
stored on the function whose code-generator needs it, rather than centrally.

Another factor may be: since `runtime::Module` are responsible for their own serialization,
and passing `tvm::Node` across `PackedFunc` requires a cast, the lack of a centralized facility for
`runtime::Modules` to obtain module-level Metadata has led backend authors to roll their own. This
pattern means that it's very difficult to assess the full scope of metadata handed to the runtime,
particularly across all backends.

This RFC argues for creating a centralized `tvm::runtime::metadata::Metadata` struct which contains
all Metadata consumed at runtime. Unifying runtime Metadata allows us to reduce the amount of
serialization logic and eliminate duplication of metadata. The current compiler design stores
centrally-produced Metadata in a side channel, but this could be improved in future RFCs e.g. should
we move away from splitting IRModules per backend.

This RFC argues for a restructuring of the way we export Metadata through the following steps:

1. Rename `runtime::MetadataModule` to `runtime::ConstLoaderModule` to disambiguate the two and make
   its purpose in life clearer.
2. Expand the function metadata in the existing `relay::backend::ExecutorCodegenMetadata` to parity with
   `runtime::FunctionInfo`, plus include `_sizes` from `tvm::relay::transform::FunctionInfoNode` and
   the required `shape` and `dtype` information from the beginning of this section.
3. Introduce `ModelMetadataModule` to contain this information for use with the C++ runtime.

    ```bash
    class ModelMetadataModule {
      virtual GetFunction(const std::string& name, ObjectPtr<Object>& sptr_to_self) {
        if (name == "get_model_metadata") {
           return PackedFunc([](TVMArgs args, TVMRetValue* rv) {
              *rv = ModelMetadata(metadata_);
           });
        } else {
          return PackedFunc();
        }
      }

      const struct ModelMetadata* metadata_;
    };
    ```

4. Introduce an optional implementation for the C runtime.
5. Export runtime::Metadata to Model Library Format.

The new proposed definition of `runtime::Metadata` is as follows.  NOTE that this is a C definition
because it will be made available both the C and C++ runtimes. A C++ wrapper will be written.

```bash
struct ParameterInfo {
  const char* relay_name_hint;
  const char* tir_name_hint;
  int64_t* shape;
  int64_t ndim;
  DLDataType dtype;
  TargetDevice target_device;  // NOTE: future addition; not covered in this RFC.
};

struct FunctionInfo {
  const char* function_name;
  struct ParameterInfo* params;
  int num_inputs;
  int num_outputs;
  int64_t workspace_size_bytes;
  int64_t io_size_bytes;
  int64_t constant_size_bytes;
};

typedef struct Metadata {
  int version;
  struct FunctionInfo* functions;
  const char* module_name;
};
```

### Internal workings of AotExecutor (`--runtime=c++ --interface-api=packed`)

Given the above, we can now sketch out the way AotExecutor should behave (for C++ runtime).

Module initialization will:

1. Load the `ModelMetadata` using `get_model_metadata` PackedFunc.
2. Allocate space for the parameters to `tvmgen_<model_name>_run_model`.
3. Lookup and load any linked parameters using the `--link-params` mechanism.

- `set_input`, `get_input`, `get_output` all work as they do in `GraphExecutor`.
- `run` assembles `TVMArgs` containing inputs + outputs and invokes `tvmgen_<model_name>_run_model`.
- `time_evaluator` is implemented in the same way as it is in `GraphExecutor`. Timing `run_model` is
  done using the CPU timer.

### Internal workings of AotExecutor (`--runtime=c --interface-api=packed`)

The C runtime version works in a very similar way with C accessor functions for the `ModelMetadata`.

### No AotExecutor implementation planned (`--runtime=c --interface-api=c`)

When `-interface-api=c` is present in the Target string, the `run_model` function no longer accepts
the PackedFunc interface and instead accepts `arg_values` directly as positional args:

```bash
TVM_DLL int32_t tvmgen_default_run_model(void* arg0, void* arg1, void* arg2) {
  void* input = arg0;
  void* input1 = arg1;
  void* output = arg2;
  (void)tvmgen_default_fused_multiply(input, input1, output);
  return 0;
}
```

Additional work is underway to wrap this in a firmware-friendly interface. A core design goal of
this interface is to offload all memory management tasks to the calling code to facilitate
integration with bare-metal embedded devices.

Therefore, it would go against the goals of the C interface to introduce a generic runtime wrapper
compatible with PackedFunc calling convention. It may be possible to do so in the future, but it
would be great to motivate such an implementation with rationale more related to the embedded
runtime setting.

### Operator Calling Convention

TVM uses 3 internal calling conventions:

1. `call_packed` - the traditional calling convention used in the C++ runtime
2. `call_cpacked` - similar to `call_packed`, but TVM presumes a symbol is linked into the binary
   containing that function name (e.g. `TVMBackendGetFuncFromEnv` is not used to lookup the
   PackedFunc)
3. `unpacked` - used with microTVM to avoid overhead of PackedFunc calls in statically-linked
   binaries. See [AOT optimisations for Embedded Targets
   RFC](https://discuss.tvm.apache.org/t/rfc-utvm-aot-optimisations-for-embedded-targets/9849).

The AOT `run_func` can use a different calling convention externally (e.g. `--interface-api`) than
that used internally with Implemented Operators (`--unpacked-args`). However, there are some
circumstances under which not all choices can be used:

- When targeting the C++ runtime: `call_packed` must be used when non-DSO-exportable modules exist;
  otherwise `call_cpacked` may be used. `unpacked` may not be used with AOT Executor as the
  interface has not settled.
- When targeting the C runtime: any calling convention may be selected for either the interface API
  or the operator calling convention. However, when using `--interface-api=c` (e.g. `unpacked`
  `run_func` calling convention), you must also use the `unpacked` calling convention with
  Implemented Operators.

# **Drawbacks**

Why should we  *not*  do this?

- This requires quite a bit of rework of the Metadata-passing mechanism, with potential for breakage.
- It also introduces yet another Executor to the runtime to maintain.
- It may introduce additional constraints on the `<C-runtime, C-interface>` implementation, which
  may make it more difficult to make progress on microTVM.

# **Rationale and alternatives**

- Why is this design the best in the space of possible designs?
- What other designs have been considered and what is the rationale for not choosing them?
- What is the impact of not doing this?

This RFC doesn't address the question of "why add an AOT executor?" The RFC which added it in the
first place is a better location to look for rationale to motivate that. In general, not following
through with this RFC would relegate the AOT executor to a C-runtime-only component. There is
significant interest in AOT from C++ runtime users, and maintaining compatibility with both
increases the chances that AOT executor will support all TVM runtime features.

The controversial pieces of this RFC addressed are as follows:

### Should we maintain a unified approach to code-generating the AOT executor?

An alternative approach could introduce an additional e.g. `aot_cpp_executor_codegen.cc` and create
a third pathway (in the Graph/AOT build flow). Doing this allows us to implement runtime-specific
compiler primitives, which may simplify both pipelines. However, soon those pipelines will grow more
complicated as features are added to leverage AOT, such as Unified Static Memory Planning. The
burden of double-maintenance of those features outweighs the advantage of a simplified
implementation. It also makes it easier for newcomers to understand the compiler.

### Should we attempt to unify the Metadata?

Metadata could be left in the scattered form it is now. It may be that the implementation of this
RFC prioritizes expansion of `ModelMetadata` over propagating it to the various non-DSO-exportable
`runtime::Module`. Ultimately though, maintaining separate function-level metadata adds confusion
and code bloat. It also makes it harder to reason about the compiler as a whole. For these reasons,
this RFC advocates for centralizing the Metadata.

# **Prior art**

There is no known prior art of a C++-runtime-compatible AOT implementation.

# **Unresolved questions**

- Who will we break if we unify Model metadata?
- Will this play nicely with the VM compilation flow when it is unified?
- How will TargetDevice come in to play here?

# **Future possibilities**

Not covered in this RFC, but particularly useful with the C++ runtime, is heterogenous execution. In
the present PoC, AotExecutor will CHECK-fail if a non-cpu device is given. A future implementation
will annotate the parameters with one of:

- A `device_type` — in which case mapping from `device_type` to `tvm::Device` will be done in the
  same way as the `GraphExecutor`
- A `target_device` — in which case a new mapping will be defined

Aside from that, the larger unresolved bit which makes it difficult to add heterogenous execution is:

- How should AOT codegen invoke the Device API?

Before this question can be answered, some progress needs to be made on the [C device
API](https://discuss.tvm.apache.org/t/pre-rfc-c-device-api/10874) and we need to define TIR
bindings.
