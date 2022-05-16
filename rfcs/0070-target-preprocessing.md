- Feature Name: target-architecture-preprocessor
- Start Date: 2022-04-04
- RFC PR: [apache/tvm-rfcs#0070](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary
Provide a standard and easily testable way to inspect architecture extensions and provide them to the various parts of TVM which utilise that information.

# Motivation
[motivation]: #motivation
TVM has multiple ways to define a `Target`s architectural features for use in deciding on schedules or other calculations, here's a few different ways we do this:

* CPU to Feature Mapping: https://github.com/apache/tvm/blob/d2db9cb0d839e32778f461b77e59f6418282a511/python/tvm/target/arm_isa.py#L22-L39
* Inspecting `Target` in utility functions: https://github.com/apache/tvm/blob/d2db9cb0d839e32778f461b77e59f6418282a511/python/tvm/topi/arm_cpu/arm_utils.py#L24-L70
* Inspecting `Target` in utility functions inside legalization code: https://github.com/apache/tvm/blob/02fbaf0ed9120a8f95155e63de42459f230584aa/python/tvm/relay/qnn/op/legalizations.py#L350-L359
* Inspecting `Target` inside the definition a strategy: https://github.com/apache/tvm/blob/b542724873140bb051492530d97a78b9b7b7983d/python/tvm/relay/op/strategy/arm_cpu.py#L232
* Processing bespoke Compiler arguments: https://github.com/apache/tvm/blob/d2db9cb0d839e32778f461b77e59f6418282a511/src/relay/backend/contrib/cmsisnn/compiler_attrs.cc#L47-L70
* Registered as a `PackedFunc` (https://github.com/apache/tvm/blob/24e5498021cecca2fe7d44149ce90efe28b6d930/python/tvm/topi/x86/utils.py#L21-L34) and then used as part of `Op` processing: https://github.com/apache/tvm/blob/24e5498021cecca2fe7d44149ce90efe28b6d930/src/relay/qnn/op/requantize_config.h#L58-L73

This RFC aims to standardise the way in which we convert `Target` attributes into architectural features by processing them ahead of time.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Two additional pre-processors can be added to the `Target`, for users to preprocess architectural information when the `Target` is created:
* Architecture Pre-processing - maps `Target` `attrs` to a new `arch` object
* Keys Pre-processing - maps `Target` `attrs` and `keys` to a new set of `keys`

These new preprocessors will be illustrated using examples targeting TVM for Arm(R) Cortex(R)-M4.

## Architecture Pre-processing
```c++
TVM_REGISTER_TARGET_KIND("c", kDLCPU)
    .set_arch_preprocessor(MyArchPreprocessor)
```

This takes the `attrs` from `Target` and converts them into an object representing the architectural features of the `Target`, which can then be accessed using the `GetArch` method similar to `GetAttr`:

```c++
Target my_target("c -mcpu=cortex-m4");
my_target->GetArch<Bool>("is_aarch64", false); // false
my_target->GetArch<Bool>("has_dsp", false); // true
```

```python
my_target = Target("c -mcpu=cortex-m4")
my_target.arch.is_aarch64 // false
my_target.arch.has_dsp // true
```

## Keys Pre-processing

```c++
TVM_REGISTER_TARGET_KIND("c", kDLCPU)
    .set_keys_preprocessor(MyKeysPreprocessor)
```

This takes the `attrs` from `Target` and maps them to relevant `keys` for use when selecting schedules:

```c++
Target my_target("c -mcpu=cortex-m4");
my_target->keys; // ["arm_cpu", "cpu"] <-- "cpu" is taken from default keys and merged by the pre-preprocessor
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Currently, there is a single `preprocessor` which takes an input of `attrs` and expects the same `attrs` returned with pre-processing applied:

https://github.com/apache/tvm/blob/d2db9cb0d839e32778f461b77e59f6418282a511/src/target/target.cc#L810-L814

In extension to this, a series of new pre-processors will be defined:

```c++
using TargetAttrs = Map<String, ObjectRef>;
using TargetArch = Map<String, ObjectRef>;
using TargetKeys = Array<String>;

using FTVMAttrPreprocessor = runtime::TypedPackedFunc<TargetAttrs(TargetAttrs)>;
using FTVMArchPreprocessor = runtime::TypedPackedFunc<TargetArch(TargetAttrs)>;
using FTVMKeysPreprocessor = runtime::TypedPackedFunc<TargetKeys(TargetAttrs, TargetKeys)>;
```

These implementations can be stored under `src/target/preprocessors/<arch_identifier>.{cc.h}` to allow them to be composed together such as:

* src/target/preprocessors/aarch64.cc
* src/target/preprocessors/cpu.cc

Where the `cpu` pre-processor can utilise the `aarch64` pre-processor if detected.

## Rename Attr Preprocessor
To help avoid confusion between the existing `attrs` `preprocessor` and the new pre-processors, the `attrs` pre-processor will be renamed from `preprocessor` to `attr_preprocessor`:

```c++
class TargetKind {
    ...
    FTVMAttrPreprocessor attr_preprocessor;

    ...
}
```

## Architecture Preprocessor
The first new pre-processor, which processes `attrs` in to an `arch` object, is registered as a new field is added to `TargetKind`:

```c++
class TargetKind {
    ...
    FTVMArchPreprocessor arch_preprocessor;

    ...
}
```

This pre-processes `Target` attributes into a new field on `Target` called `arch`:
```c++
class Target {
    ...
    DictAttrs arch;
    
    ...
}
```

Which will have similar helper methods to those seen in `IRModule` for `DictAttrs` but with reference to `Arch` rather than `Attr`:

```c++
template <typename TObjectRef>
Optional<TObjectRef> GetArch(
    const std::string& attr_key,
    Optional<TObjectRef> default_value = Optional<TObjectRef>(nullptr)) const {
return attrs.GetAttr(attr_key, default_value);
}
template <typename TObjectRef>
Optional<TObjectRef> GetArch(const std::string& attr_key, TObjectRef default_value) const {
return GetArch<TObjectRef>(attr_key, Optional<TObjectRef>(default_value));
}
```

As well as a Python class to represent this and allow simple access:
```python
class TargetArch {
    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        return _ffi_api.TargetGetArch(self._target, name)
}
```

## Key Preprocessor
The second new pre-processor will populate the `keys` fields from the initial `Target` `attrs` and existing `keys`, it simply requires an additional field on `TargetKind`:

```c++
class TargetKind {
    ...
    FTVMKeysPreprocessor keys_preprocessor;

    ...
}
```

As the signature of the pre-processor passes the existing keys into the `keys_preprocessor` it is responsible for merging them or removing them if necessary.

# Drawbacks
[drawbacks]: #drawbacks

By adding these new pre-processing options to `Target` we increase the amount of work incurred when instantiating a `Target`, it was ultimately considered that this one-time cost would be similar to repeatedly querying the `Target` attributes.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

## Re-use Target Attributes
Instead of creating a new field (`arch`), instead extend `Target` attributes with preprocessed results such that you can do:

```python
my_target = Target("c -mcpu=cortex-m4")
my_target.is_aarch64 # Extra attribute in `attrs`
```

It was felt this would become confusing to work with alongside the documented `Target` attributes in `target_kind.cc` or `target_kind.cc` would need to be bloated with every potential architecture field. The approach of overlapping with `Target` attributes would also increase testing overhead rather than having a straight forward `attrs` to `arch` mapping to test.

## Extend Utility Functions
Using a standalone function or class across the various areas of the codebase, such as:

```
TargetArch my_target_arch(target)
my_target_arch->is_aarch64; // false
```

This means re-processing `Target` whenever a specific attribute is required but would provide a single source of truth for doing so.

# Prior art
[prior-art]: #prior-art

Taking the example of LLVM, it follows a similar methodology, resulting in a `Features` vector:
* `clang` uses `mtriple` to determine the correct parser to use for the various other options: https://github.com/llvm/llvm-project/blob/2f04e703bff3d9858f53225fa7c780b240c3e247/clang/lib/Driver/ToolChains/Clang.cpp#L324
* `clang` uses the LLVM parsers to determine available features for a given set of `Target` parameters such as `mcpu` and `mtune`: https://github.com/llvm/llvm-project/blob/43d758b142bbdf94a1c55dc0950637ae74f825b9/clang/lib/Driver/ToolChains/Arch/AArch64.cpp
* LLVM implements the `Features` parsers: https://github.com/llvm/llvm-project/blob/09c2b7c35af8c4bad39f03e9f60df8bd07323028/llvm/lib/Support/AArch64TargetParser.cpp
* The parser is tested in insolation: https://github.com/llvm/llvm-project/blob/09c2b7c35af8c4bad39f03e9f60df8bd07323028/llvm/unittests/Support/TargetParserTest.cpp

You can see similar definitions within GCC: 
* Pre-processes the CLI arguments to add more specific flags: https://github.com/gcc-mirror/gcc/blob/16e2427f50c208dfe07d07f18009969502c25dc8/gcc/config/aarch64/driver-aarch64.c#L246
* Extensions are defined here: https://github.com/gcc-mirror/gcc/blob/16e2427f50c208dfe07d07f18009969502c25dc8/gcc/config/aarch64/aarch64-option-extensions.def

This RFC builds upon the following existing TVM RFCs:
* This follows the original Target Specification RFC: https://discuss.tvm.apache.org/t/rfc-tvm-target-specification/6844
* Pre-processor definitions follow the pattern set out in Target Hooks: https://github.com/apache/tvm-rfcs/blob/main/rfcs/0010-target-registered-compiler-flow-customisation.md

# Unresolved questions
[unresolved-questions]: #unresolved-questions

# Future possibilities
[future-possibilities]: #future-possibilities

Similar to LLVM and GCC, we may be able to use a custom file format to describe `Target`s more effectively in future which can be added using the same hooks, allowing for easier contributions.
