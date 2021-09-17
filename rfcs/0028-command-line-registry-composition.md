- Feature Name: Command Line Composition from Internal Registry
- Start Date: 2021-08-24
- RFC PR: [apache/tvm-rfcs#28](https://github.com/apache/tvm-rfcs/pull/28)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

Introducing a standardised form for `tvmc` arguments to be populated from internal registries in TVM.

# Motivation
[motivation]: #motivation

Currently, when a user uses `tvmc`, they present a target string:
```
tvmc --target="woofles -mcpu=woof, c -mattr=+mwoof"
```

Using a target string here means that the entire `--target` argument is used as an opaque pass-through to the internal `Target` string parser. Users coming to TVM should be able to compose these options and get meaningful help when doing that composition with `tvmc`. As an example, using the default `argparse` behaviour, the arguments can be registered as follows:
```python
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--target', type=str, help='comma separated target list or target string')

target_c = parser.add_argument_group('target c')
target_c.add_argument('--target-c-mcpu', type=str, help='target c mcpu string')
target_c.add_argument('--target-c-mattr', type=str, help='target c mattr string')

target_llvm = parser.add_argument_group('target llvm')
target_llvm.add_argument('--target-llvm-mcpu', type=str, help='target llvm mcpu string')
target_llvm.add_argument('--target-llvm-mattr', type=str, help='target lvm mattr string')

args = parser.parse_args()
print(args)
```

The user can get help for all available target options:
```
usage: test.py [-h] [--target TARGET] [--target-c-mcpu TARGET_C_MCPU] [--target-c-mattr TARGET_C_MATTR] [--target-llvm-mcpu TARGET_LLVM_MCPU] [--target-llvm-mattr TARGET_LLVM_MATTR]

optional arguments:
  -h, --help            show this help message and exit
  --target TARGET       comma separated target list or target string (default: None)

target c:
  --target-c-mcpu TARGET_C_MCPU
                        target c mcpu string (default: None)
  --target-c-mattr TARGET_C_MATTR
                        target c mattr string (default: None)

target llvm:
  --target-llvm-mcpu TARGET_LLVM_MCPU
                        target llvm mcpu string (default: None)
  --target-llvm-mattr TARGET_LLVM_MATTR
                        target lvm mattr string (default: None)
```

These are arranged in per-`Target` groups to allow the user to easily follow which are applicable and which aren't. The use can now replace their existing `Target` string, by using the documented `Target` options, with:
```
tvmc --target=woofles,c \
    --target-woofles-mcpu=woof \
    --target-c-mattr=+mwoof
```

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Alongside support for a target string, `tvmc` would populate a series of other arguments specific to targets which match the attributes found in the TargetKindRegistry in `target_kind.cc`. For example, consider a subset of the `c` `Target`:

```
TVM_REGISTER_TARGET_KIND("c", kDLCPU)
    .add_attr_option<String>("mcpu")
```

This would be translated at the `tvmc` level to:
```bash
tvmc --target=c \
    --target-c-mcpu=cortex-m3
```

Which would then allow the user to compose together these options, such as:

```bash
tvmc --target=cmsisnn,c \ # Specifying multiple targets to enable in priority order
    --target-cmsisnn-mattr=+dsp \
    --target-c-mcpu=cortex-m3
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

There already exists a mechanism which provides Python with `PassConfig` information, via:

```c++
// Function Registry
TVM_REGISTER_GLOBAL("transform.ListConfigs").set_body_typed(PassContext::ListConfigs);
// Actual call
Map<String, Map<String, String>> PassContext::ListConfigs() {
  return PassConfigManager::Global()->ListConfigs();
}
// Implementation
Map<String, Map<String, String>> ListConfigs() {
    Map<String, Map<String, String>> configs;
    for (const auto& kv : key2vtype_) {
        Map<String, String> metadata;
        metadata.Set("type", kv.second.type_key);
        configs.Set(kv.first, metadata);
    }
    return configs;
}
```

This can be replicated to provide the same information for `TargetKind` or any other registry and provide this information to generate arguments in `tvmc` using `argparse`:
```
parser.add_argument(f"--target-{kind}-{attr}", ...)
```

# Drawbacks
[drawbacks]: #drawbacks

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

There's a number of alternative methods for injecting this information which have been used in other CLIs but obfuscate some of the information from the CLI arguments themselves:
- Continue using target strings
- Specify JSON

# Prior art
[prior-art]: #prior-art

This is not dissimilar to how `gcc` or `clang` provide options, [for example in gcc](https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html):
```
gcc -fsanitize=address -fsanitize-address-use-after-scope
```

# Unresolved questions
[unresolved-questions]: #unresolved-questions



# Future possibilities
[future-possibilities]: #future-possibilities

## Pattern Re-use
By creating the infrastructure to allow this composition and providing a standard pattern, we can allow `tvmc` users to compose other aspects of TVM in a similar way, such as `executor`:

```
tvmc --target=cmsisnn,c \ # Specifying multiple targets to enable in priority order
    --target-cmsisnn-mattr=+dsp \
    --target-c-mcpu=cortex-m3 \
    --executor=aot \
    --executor-aot-unpacked-api=1
```