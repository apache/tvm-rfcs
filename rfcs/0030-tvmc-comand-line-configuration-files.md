- Feature Name: Command Line Configuration Files
- Start Date: 2021-08-09
- RFC PR: [apache/tvm-rfcs#30](https://github.com/apache/tvm-rfcs/pull/30)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

Collecting common configurations for users of TVM and exposing them gracefully in `tvmc` using a `--config` option. The scope of this RFC is to introducing the configuration files, the placement of them and demonstrating usage.

# Motivation
[motivation]: #motivation

When a user first approaches TVM, choosing an appropriate configuration can be difficult, this is increasingly true in embedded systems where the configuration is not only a collection of devices but also how those devices are interfaced (see [Arm&reg; Corstone&trade;-300 reference package](https://developer.arm.com/ip-products/subsystem/corstone/corstone-300)). Trying to specify all of this in a target string or via command line arguments would be error prone and tedious. Predefining these in a common format allows users to get started and take the configurations for their own use cases easily.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

## TVM Hosted Configurations
Configurations will be stored as [JSON5](https://json5.org/) at `configs/<TYPE>/<NAME>.json`, this top level directory will enable other tooling to load configurations just as easily a `tvmc` and provide easy sign posting for users looking for configurations. This folder structure includes two levels to allow contributors of configurations to choose appropriate values for:
* `<TYPE>` - A suitable collective under which configurations can live, in this document the example used is `boards` but this could equally be `instances` for a cloud provider and is unbounded to allow contributors to group in the most effective way.
* `<NAME>` - The name of the configuration, such as a board name or other composite structure of configurations.

A user coming to `tvmc` will begin with a default configuration which sets sensible defaults, such that `tvmc compile my_model.tflite` works out of the box. This is enabled by a `configs/host/default.json` which is likely to specify:

```
{
  targets: [
    {
      kind: "llvm"
    }
  ]
}
```

As a more substantial example, you can imagine an embedded board configuration such as the [Arm&reg; Corstone&trade;-300 reference package](https://developer.arm.com/ip-products/subsystem/corstone/corstone-300), which would exist under `configs/boards/corstone-300.json`:

```json
{
  "output_format": "mlf",
  "executor": {
    "kind": "aot",
    "unpacked-api": true
  },
  "targets": [
    {
      "kind": "ethos-u",
      "accelerator_config": "ethos-u55-32"
    },
    {
      "kind": "cmsisnn",
      "mattr": "+fp"
    },
    {
      "kind": "llvm"
    }
  ]
}
```

This would be used if the user simply specifies `--config=corstone-300`, as in the following example:
```
tvmc compile --config=corstone-300 my_model.tflite
```

## User Provided Configurations
The default search path, as illustrated above, is to find a matching `<NAME>.json` to an argument `--config=<NAME>`. A user can instead specify a path in the `--config` argument such as:

```bash
--config=./my.json
--config=/etc/devices/my_secret_board.json
```

By default, TVM will prefer files explicitly specified as a path instead of hosted files.

## Combination with existing parameters
In the case of `tvmc`, `--config` will work alongside other arguments. Ideally anything specifiable in JSON will be specifiable in the command line to allow users to make small alterations such as:

```
tvmc \
  compile \
  --config=corstone-300 \
  --executor-aot-unpacked-api=0
```

Which allows experimentation with different parameters that can then be added to a custom JSON. Complex configurations which aren't easily represented well in CLI arguments may exist and can continue to be represented only in JSON.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

To get the `--config` flag, `argparse` can be used as an early pass over the arguments to collect the single configuration file to specify. It's important to note that only one configuration file would be supported per command line and the default `argparse` behaviour of taking the last `--config` flag would take effect.

This will change the behaviour of how `tvmc` utilises `argparse`, it will first translate arguments from `argparse` into an internal dictionary of attributes and then apply those over the top of any specified configuration files. This means the default options for `argparse` are essentially nulled as they won't be aware of configuration files until after the arguments are parsed. The hierarchy is therefore:
1. Arguments parsed by `argparse`
2. Configuration file specified (defaults to `default`)
3. Internal defaults for arguments in `tvmc`

## Example: merging with new config
This example is using the changes illustrated in [Migrating Target Attributes to IRModule](https://github.com/apache/tvm-rfcs/pull/29) to provide a clearer example of how extension occurs. If [Migrating Target Attributes to IRModule](https://github.com/apache/tvm-rfcs/pull/29) is eventually rejected, this example serves to demonstrate the logic with which merging occurs and could be applied to a variety of configuration combinations. 

Because these results are merged, the underlying defaults remain in `tvmc` rather than in `default.json` to ensure the user doesn't create a resulting configuration which additively makes no sense (for example, being based on an `llvm` target or other defaults). For example, the default in `tvmc` would be:
```json
{ "autotuning_runs": 10 }
```
Which would then be extended with `--config=default` (`{ "targets": [{ "kind": "llvm" }], "executor": { "kind": "graph", "system-lib": true } }`):
```json
{ "autotuning_runs": 10, "targets": [{ "kind": "llvm" }], "executor": { "kind": "graph", "system-lib": true } }
```
**Or** be extended by `--config=corstone300` (`{ "targets": [{ "kind": "c", "mcpu": "cortex-m55" }, { "kind": "ethosu" }] }`):
```json
{ "autotuning_runs": 10, "targets": [{ "kind": "c", "mcpu": "cortex-m55" }, { "kind": "ethosu" }] }
```
This can then be further overrided by the CLI `--config=corstone300 --target=llvm --target-llvm-mattr=+fp`:
```json
{ "autotuning_runs": 10, "targets": [{ "kind": "llvm", "mattr": "+fp" }] }
```
**Or** overriding specific `Target` options (`--config=corstone300 --target-c-mcpu=cortex-m4`):
```json
{ "autotuning_runs": 10, "targets": [{ "kind": "c", "mcpu": "cortex-m4" }, { "kind": "ethosu" }] }
```

It can be seen that this merging follows a simple algorithm:
1. If the config file specifies `targets` key, all `targets` from the default config are deleted/overridden.
2. If the command-line supplies `--target=`, all `targets` from the config file are deleted/overridden.
3. If the command-line only supplies e.g. `--target-llvm-mcpu`, then it modifies the llvm target from config/defaults.
4. If the command-line or a config file specifies a `target` sub-key, it always overrides it in full (e.g. there is no appending to mattr from the command line).

Notably this algorithm will apply across all registries uniformly once [Command Line Composition from Internal Registry](https://github.com/apache/tvm-rfcs/pull/28) lands.

## Example: merging on top of default.json
The undesirable behaviour would be something such as, this default in `tvmc`:
```json
{ "autotuning_runs": 10 }
```
Which would then be extended with `--config=default` (`{ "targets": [{ "kind": "llvm" }], "executor": { "kind": "aot", "system-lib": true } }`):
```json
{ "autotuning_runs": 10, "targets": [{ "kind": "llvm" }], "executor": { "kind": "aot", "system-lib": true } }
```
And further extended on top of `default.json` with `--config=woofles` (`{ "targets": [{ "kind": "llvm" }], "executor": { "kind": "aot", "unpacked-api": true } }`):
```json
{ "autotuning_runs": 10, "targets": [{ "kind": "llvm" }], "executor": { "kind": "aot", "unpacked-api": true, "system-lib": true } }
```
We've now acquired undesirable arguments (`"system-lib": true`) which we would not want passed to the AOT executor for this platform, this is due to:
1. The `default.json` config file specifies the `executor` key, this sets up a default excecutor
2. The `woofles.json` config file specifies the `system-lib` key on the `executor`, which adds the property on top of existing properties

## Configuration format
The configuration files will be loaded using [json5](https://pypi.org/project/json5/) to enable us to add comments and further details to the JSON files. JSON5 extends upon JSON to provide for comments and other documentation features for users.

## Configuration schema
Configuration file schema will be maintained in `configs/schema.json` in [JSON Schema format](https://pypi.org/project/jsonschema/).

# Drawbacks
[drawbacks]: #drawbacks

Although this presents a simpler interface for new users, the options become more complex for power users who want to mix and match arguments. This is mitigated by specifying simple rules for how these are composed.

The convention of `--config` being read first then all other arguments is less intuitive than left to right parsing, but fits better with the current `argparse` infrastructure.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

Other configuration formats were considered but JSON5 was selected as a reasonably structured and easy to understand format which is widely used for configuration. Specifically YAML was considered but has looser structure than JSON which leaves more space for user error and JSON is already prevalent in TVM.

# Prior art
[prior-art]: #prior-art

Such configuration files already exist in a number of platforms and tools to reduce the overhead on the user to define them, such as the [Zepyhr DeviceTree](https://github.com/zephyrproject-rtos/zephyr/tree/main/boards) or [gcc configurations for specific targets](https://github.com/gcc-mirror/gcc/blob/16e2427f50c208dfe07d07f18009969502c25dc8/gcc/config/arm/arm-cpus.in).

The configuration files in JSON format lends itself to following the structure similar to that proposed in [TVM Target Specification](https://discuss.tvm.apache.org/t/rfc-tvm-target-specification/6844).

# Unresolved questions
[unresolved-questions]: #unresolved-questions

# Future possibilities
[future-possibilities]: #future-possibilities

By starting to map arguments between a configuration file as well as command line arguments they should start to align with standard rule sets. These rule sets can be used to then augment the CLI args and configuration files with a further option of environment variables - for an example of this see [Terraform](https://www.terraform.io/docs/language/values/variables.html#environment-variables).