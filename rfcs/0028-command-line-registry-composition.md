- Feature Name: Command Line Composition from Internal Registry
- Start Date: 2021-08-24
- RFC PR: [apache/tvm-rfcs#28](https://github.com/apache/tvm-rfcs/pull/28)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

Introducing a standardised form for `tvmc` arguments to be populated from internal registries in TVM. This is currently limited to just the `Target` registry but with future work, such as [Migrating Target Attributes to IRModule](https://github.com/apache/tvm-rfcs/pull/29) there will be further options that require a standard CLI mechanism to interact with. The scope of this RFC is to define the pattern for translating between these internal registries and the CLI in a standard way which can be understood by typical CLI users.

# Motivation
[motivation]: #motivation

In order to motivate this, the `Target` will be used, as mentioned above this is a singular example of a general pattern - a core tenet of good UX is a series of well understood patterns that the user can re-apply through-out the use of a product.

Coming to `tvmc` as a user 6-7 months ago, the `--target` is essentially magic compared to other CLI arguments which clearly detail the options available. Using a target string means that the entire `--target` argument is used as an opaque pass-through to the internal `Target` string parser, this obfuscates the options available to a `Target` and creates a non-standard sub-syntax for CLI users:
```
tvmc --target="woofles -mcpu=woof, c -mattr=+mwoof"
```

Users coming to TVM should be able to compose the available `Target` options and get meaningful help when doing that composition with `tvmc`, Such as:

```bash
$ tvmc --help
usage: tvmc [-h] [--target TARGET] [--target-c-mcpu TARGET_C_MCPU] [--target-c-mattr TARGET_C_MATTR] [--target-llvm-mcpu TARGET_LLVM_MCPU] [--target-llvm-mattr TARGET_LLVM_MATTR]

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

The options are arranged in per-`Target` groups to allow the user to easily follow which are applicable and which aren't. The user can now replace their existing `Target` string, by using the documented `Target` options, with:
```bash
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

This would be translated at the `tvmc` level to `argparse` parameters. As an example, using the default `argparse` behaviour, the arguments can be registered as follows:
```python
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--target', type=str, help='comma separated target list or target string')

target_c = parser.add_argument_group('target c')
target_c.add_argument('--target-c-mcpu', type=str, help='target c mcpu string')
target_c.add_argument('--target-c-mattr', type=str, help='target c mattr string')

target_cmsisnn = parser.add_argument_group('target cmsisnn')
target_cmsisnn.add_argument('--target-cmsisnn-mattr', type=str, help='target cmsisnn mattr string')

args = parser.parse_args()
print(args)
```

The user can get help for all available target options:
```
usage: tvmc [-h] [--target TARGET] [--target-c-mcpu TARGET_C_MCPU] [--target-c-mattr TARGET_C_MATTR] [--target-cmsisnn-mcpu TARGET_CMSISNN_MCPU] [--target-cmsisnn-mattr TARGET_CMSISNN_MATTR]

optional arguments:
  -h, --help            show this help message and exit
  --target TARGET       comma separated target list or target string (default: None)

target c:
  --target-c-mcpu TARGET_C_MCPU
                        target c mcpu string (default: None)
  --target-c-mattr TARGET_C_MATTR
                        target c mattr string (default: None)

target cmsisnn:
  --target-cmsisnn-mattr TARGET_CMSISNN_MATTR
                        target lvm mattr string (default: None)
```

Which would then allow the user to compose together these options, such as:

```bash
tvmc --target=cmsisnn,c \ # Specifying multiple targets to enable in priority order
    --target-cmsisnn-mattr=+dsp \
    --target-c-mcpu=cortex-m3
```

If a user tries to use an attribute of an unspecified `Target`, it is expected that it would error and provide feedback to the user, for example:

```bash
$ tvmc --target=cmsisnn,c \ 
    --target-cmsisnn-mattr=+dsp \
    --target-llvm-mcpu=cortex-m3 # Oh no, we asked for C but tried to configure LLVM
Error: target llvm mcpu passed but target llvm not specified
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
```python
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
```shell
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