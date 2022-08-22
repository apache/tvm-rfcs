- Feature Name: target-features
- Start Date: 2022-04-04
- RFC PR: [apache/tvm-rfcs#78](https://github.com/apache/tvm-rfcs/pull/78)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary
Provide a standard and easily testable way to inspect features of a given target and provide them to the various parts of TVM which utilise that information.

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

An additional property `features` will be added to the `Target` which is created at the time of instantiation, this will be populated by inferred features of the `Target` such as architectural extensions or bus sizes. The main distinction is that `features` are inferred from the `Target` `attrs` rather than being passed in.

An example of the new `features` attribute will be illustrated using examples targeting TVM for Arm(R) Cortex(R)-M4.

The `Target` specifies the specific CPU in the `attrs` and uses that to create the `features` object representing the architectural extensions of the `Target`, which can then be accessed using the `GetFeature` method similar to `GetAttr`:

```c++
Target my_target("c -mcpu=cortex-m4");
my_target->GetFeature<Bool>("is_aarch64", false); // false
my_target->GetFeature<Bool>("has_dsp", false); // true
```

```python
my_target = Target("c -mcpu=cortex-m4")
my_target.features.is_aarch64 # false
my_target.features.has_dsp # true
```

This means that instead of the current:

```python
isa = arm_isa.IsaAnalyzer(target)
if isa.has_dsp_support:
    do_dsp_stuff()
```

The `Target` can be directly inspected:

```python
if target.features.dsp:
    do_dsp_stuff()
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

The `Target` class, in C++, will have an an additional property named `features`:

```c++
class Target {
    ...
    DictAttrs features;
    
    ...
}
```

Which will have similar helper methods to those seen in `IRModule` for `DictAttrs` but with reference to `Features` rather than `Attr`:

```c++
template <typename TObjectRef>
Optional<TObjectRef> GetFeatures(
    const std::string& attr_key,
    Optional<TObjectRef> default_value = Optional<TObjectRef>(nullptr)) const {
    return attrs.GetAttr(attr_key, default_value);
}

template <typename TObjectRef>
Optional<TObjectRef> GetFeatures(const std::string& attr_key, TObjectRef default_value) const {
    return GetFeatures<TObjectRef>(attr_key, Optional<TObjectRef>(default_value));
}
```

As well as a Python class to represent this and allow simple access to the `features` using the `target.features.<feature>` syntax:
```python
class TargetFeatures:
    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        return _ffi_api.TargetGetFeature(self._target, name)
```

# Drawbacks
[drawbacks]: #drawbacks

Centralising `features` on `Target` increases the complexity for each `Target` parser as they will have to cater for a number of attributes, this is easily avoided by splitting the internal parsers.

Making `features` read-only and derived from the parser limits the flexibility to create an object with specific features for testing, in this case actual valid `Target`s will have to be used for such testing.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

## Re-use Target Attributes
If we were to attach all of these directly to `Target` (i.e. `llvm`) as `attrs`, that would drastically increase the number of fields on a given `Target` and in all cases only a subset would be used - specific to a given CPU/GPU profile:

```python
my_target = Target("c -mcpu=cortex-m4")
my_target.is_aarch64 # Extra attribute in `attrs`
```

Re-using `attrs` becomes confusing to work with alongside the documented `Target` attributes in `target_kind.cc`, or `target_kind.cc` would need to be bloated with every potential feature of a `Target`. The approach of overlapping with `Target` attributes would also increase testing overhead rather than having a straight forward `attrs` to `features` map to test you would need to consider which `attrs` could validly mutate - this also introduces user confusion as `target.mcpu` is no longer the `mcpu` which they passed in. 

## Extend Utility Functions
Using a standalone function or class across the various areas of the codebase, such as:

```c++
TargetFeatures my_target_features(target)
my_target_features->is_aarch64; // false
```

This means re-processing `Target` whenever a specific attribute is required but would provide a single source of truth for doing so.

## Target Tags
It's potentially possible to recreate the functionality of `features` by populating a larger list of `Target` tags, taking the example of:

```c++
TVM_REGISTER_TARGET_TAG("raspberry-pi/4b-aarch64")
    .set_config({{"kind", String("llvm")},
                 {"mtriple", String("aarch64-linux-gnu")},
                 {"mcpu", String("cortex-a72")},
                 {"mattr", Array<String>{"+neon"}},
                 {"num-cores", Integer(4)},
                 {"host", Map<String, ObjectRef>{{"kind", String("llvm")},
                                                 {"mtriple", String("aarch64-linux-gnu")},
                                                 {"mcpu", String("cortex-a72")},
                                                 {"mattr", Array<String>{"+neon"}},
                                                 {"num-cores", Integer(4)}}}});
```

These are pre-configured `Target`s with various `mtriple`, `mcpu` and `mattr` attributes already set - once parsed these can produce a set of architecture features for subsequent steps, such as replacing this check in the operator strategy:

https://github.com/apache/tvm/blob/f88a10fb00419c51a116a63f931a98d8286b23de/python/tvm/relay/op/strategy/arm_cpu.py#L232-L245

Other tagged `Target`s will likely have the same `mattr` and `mcpu`, thus rather than trying to hand craft the permutations each time, the parser generalises inferring these `features`, augmenting tagged `Target`s.

# Prior art
[prior-art]: #prior-art

## Other Compilers
Taking the example of LLVM, it follows a similar methodology, resulting in a `Features` vector:
* `clang` uses `mtriple` to determine the correct parser to use for the various other options: https://github.com/llvm/llvm-project/blob/2f04e703bff3d9858f53225fa7c780b240c3e247/clang/lib/Driver/ToolChains/Clang.cpp#L324
* `clang` uses the LLVM parsers to determine available features for a given set of `Target` parameters such as `mcpu` and `mtune`: https://github.com/llvm/llvm-project/blob/43d758b142bbdf94a1c55dc0950637ae74f825b9/clang/lib/Driver/ToolChains/Arch/AArch64.cpp
* LLVM implements the `Features` parsers: https://github.com/llvm/llvm-project/blob/09c2b7c35af8c4bad39f03e9f60df8bd07323028/llvm/lib/Support/AArch64TargetParser.cpp
* The parser is tested in insolation: https://github.com/llvm/llvm-project/blob/09c2b7c35af8c4bad39f03e9f60df8bd07323028/llvm/unittests/Support/TargetParserTest.cpp

You can see similar definitions within GCC: 
* Pre-processes the CLI arguments to add more specific flags: https://github.com/gcc-mirror/gcc/blob/16e2427f50c208dfe07d07f18009969502c25dc8/gcc/config/aarch64/driver-aarch64.c#L246
* Extensions are defined here: https://github.com/gcc-mirror/gcc/blob/16e2427f50c208dfe07d07f18009969502c25dc8/gcc/config/aarch64/aarch64-option-extensions.def

## Existing TVM RFCs
This RFC builds upon the following existing TVM RFCs:
* This follows the original Target Specification RFC: https://discuss.tvm.apache.org/t/rfc-tvm-target-specification/6844
* Pre-processor definitions follow the pattern set out in Target Hooks: https://github.com/apache/tvm-rfcs/blob/main/rfcs/0010-target-registered-compiler-flow-customisation.md
* Target JSON Parser: https://github.com/apache/tvm-rfcs/pulls/71

# Unresolved questions
[unresolved-questions]: #unresolved-questions

# Future possibilities
[future-possibilities]: #future-possibilities

Similar to LLVM and GCC, we may be able to use a custom file format to describe `Target`s more effectively in future which can be added using the same hooks, allowing for easier contributions.
