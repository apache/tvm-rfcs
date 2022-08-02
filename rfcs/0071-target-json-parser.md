- Feature Name: target-json-preprocessor
- Start Date: 2022-04-04
- RFC PR: [apache/tvm-rfcs#0071](https://github.com/apache/tvm-rfcs/pull/71)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary
Extend the existing `TargetKind` `preprocessor` to allow preprocessing of the entire `Target` JSON representation rather than just `attrs`.

# Motivation
[motivation]: #motivation

Taking an example `Target` in JSON form:

```js
{
    "id": "cuda",
    "tag": "nvidia/tx2-cudnn",
    "keys": ["cuda", "gpu"],
    "libs": ["cudnn"],
    "target_host": {
        "id": "llvm",
        "system_lib": True,
        "mtriple": "aarch64-linux-gnu",
        "mattr": "+neon"
    }
}
```

We can see that there are additional fields which are of interest to TVM, note-ably `keys` and `libs` which we currently do not apply parsing to on `Target` instantiation. Extending the `TargetKind` `preprocessor` beyond `attrs` enables to customise parsing of the entire `Target`, enabling the values passed by the user to be used to infer other properties used during compilation.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Alongside the existing `set_attrs_preprocessor` method on `TargetKind`, there will be an alternative `set_target_parser` method to bind a `FTVMTargetParser` to the `TargetKind`. The new `FTVMTargetParser` will take precedence over the `attrs` preprocessor if present:

```c++
TVM_REGISTER_TARGET_KIND("target", kDLCPU)
    .set_target_parser(TargetParser);
```

The canonical JSON form of `Target` will be passed to the new `Target` parser and the parser will return the transformed variant in JSON form for further steps:

```c++
using TargetJSON = Map<String, ObjectRef>;
TargetJSON TargetParser(TargetJSON target) {
    // ... transforms ...
    return target;
}
```

The parser will have to be capable of handling the diversity of types of `Target` in TVM, therefore the underlying mechanism of the parser is left as an implementation detail. Using the example of pre-processing the `keys` attribute (used for detecting appropriate schedules), it can be seen how this can apply to various `Target`s. 

## TVM Target's Directly Mapping to a Backend's Target
Take the example of pre-processing `keys` (in this case using the `cuda` `Target`):
```c++
using TargetJSON = Map<String, ObjectRef>;

TargetJSON CUDAParser(TargetJSON target) {
    if (IsSuper(target)) {
        target["keys"].push_back("super_cuda");
    }
}

TVM_REGISTER_TARGET_KIND("cuda", kDLGPU)
    .set_target_parser(CUDAParser);
```

This takes the `attrs` from `Target` and maps them to relevant `keys` for use when selecting schedules:

```c++
Target my_target("cuda -msuper");
my_target->keys; // ["cuda", "gpu", "super_cuda"] <-- "cpu" and "cuda" are taken from default keys - "super_cuda" is added
```

## TVM Target's Mapping to a Backend with Multiple Target's
The previous example would work for `Target`s which map to a specific architecture, such as `cuda`. To parse a `Target` which has a number of its own targets, such as `llvm`, the parser can be broken down within the parent parser:

```c++
using TargetJSON = Map<String, ObjectRef>;

TargetJSON AArch64TargetParser(TargetJSON target) {
    target["keys"].push_back("arm_cpu");
    return target;
}

TargetJSON x86TargetParser(TargetJSON target) {
    target["keys"].push_back("x86_64");
    return target;
}

TargetJSON CPUTargetParser(TargetJSON target) {
    if (IsAArch64Target(target)) {
        return AArch64TargetParser(target);
    }
    if (IsX86Target(target)) {
        return x86TargetParser(target);
    }
    return target;
}

TVM_REGISTER_TARGET_KIND("llvm", kDLCPU)
    .set_target_parser(CPUTargetParser);
```

This has the additional advantage that if there are standard arguments, such as `mcpu`, `mattr` and `march`, the parser can be re-used in both `Target`s - for example the `c` `Target` can re-use the above `llvm` `Target` parser:

```c++
TVM_REGISTER_TARGET_KIND("c", kDLCPU)
    .set_target_parser(CPUTargetParser);
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Currently, there is a single `preprocessor` which takes an input of `attrs` and expects the same `attrs` returned with pre-processing applied:

https://github.com/apache/tvm/blob/d2db9cb0d839e32778f461b77e59f6418282a511/src/target/target.cc#L810-L814

The new `Target` parser will live in addition to the `preprocessor` until such a time as the `preprocessor` can be fully removed. This extends `TargetKind` to support both `preprocessor` and `target_parser`:

```c++
using TargetJSON = Map<String, ObjectRef>;
using FTVMTargetParser = TypedPackedFunc<TargetJSON(TargetJSON)>;

class TargetKind {
    ...
    PackedFunc preprocessor;
    FTVMTargetParser target_parser;

    ...
}
```

Implementations for `Target` parsers will be stored under `src/target/parsers/<parser_identifier>.{cc.h}`, allowing them to be composed together (as shown above), such as:

* src/target/parsers/cuda.cc
* src/target/parsers/aarch64.cc
* src/target/parsers/cpu.cc

Where the `cpu` pre-processor can utilise the `aarch64` pre-processor if detected and `cuda` is an independent parser specific to that `Target`.

# Drawbacks
[drawbacks]: #drawbacks

By adding these new pre-processing options to `Target` we increase the amount of work incurred when instantiating a `Target`, it was ultimately considered that this one-time cost would be similar or less than repeatedly querying the `Target` attributes. 

Providing the ability to completely change a `Target` on parsing could allow an extensive mutation of the input `Target`. 

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

Instead of providing a single parser entrypoint, we can instead use several parsers for each attribute - this clearly separates the responsibility of each parser but also means maintaining many entrypoints to `Target` parsing.

# Prior art
[prior-art]: #prior-art

## Other Compilers
Taking the example of LLVM, it follows a similar methodology, resulting in a `Features` vector:
* `clang` uses the LLVM parsers to determine available features for a given set of `Target` parameters such as `mcpu` and `mtune`: https://github.com/llvm/llvm-project/blob/43d758b142bbdf94a1c55dc0950637ae74f825b9/clang/lib/Driver/ToolChains/Arch/AArch64.cpp
* LLVM implements the `Features` parsers: https://github.com/llvm/llvm-project/blob/09c2b7c35af8c4bad39f03e9f60df8bd07323028/llvm/lib/Support/AArch64TargetParser.cpp
* The parser is tested in insolation: https://github.com/llvm/llvm-project/blob/09c2b7c35af8c4bad39f03e9f60df8bd07323028/llvm/unittests/Support/TargetParserTest.cpp

## Existing TVM RFCs
This RFC builds upon the following existing TVM RFCs:
* This follows the original Target Specification RFC: https://discuss.tvm.apache.org/t/rfc-tvm-target-specification/6844

# Unresolved questions
[unresolved-questions]: #unresolved-questions

# Future possibilities
[future-possibilities]: #future-possibilities
