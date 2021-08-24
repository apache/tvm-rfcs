- Feature Name: Migrating Target Attributes to IRModule
- Start Date: 2021-08-23
- RFC PR: [apache/tvm-rfcs#29](https://github.com/apache/tvm-rfcs/pull/29)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

Taking parameters which have been historically attached to `Target` and placing them as attributes on `IRModule`s to reflect their more global compiler state.

# Motivation
[motivation]: #motivation

There are a number of parameters which exist on `Target` which are not necessarily relevant to the `Target` but have become associated with a host `Target` due to a lack of a better place to put them, such as `executor` and `unpacked-api`. If we continue down this path, the `Target` will become so overloaded it'll become hard to understand for our potential user base.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

We create a series of new registries based on the attributes currently listed in `TargetKind`, taking the example of the `c` `TargetKind`:

```
TVM_REGISTER_TARGET_KIND("c", kDLCPU)
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Bool>("link-params", Bool(false))
    .add_attr_option<String>("runtime")
    .add_attr_option<String>("mcpu")
    .add_attr_option<String>("march")
    .add_attr_option<String>("executor")
    .add_attr_option<Integer>("workspace-byte-alignment")
    .add_attr_option<Bool>("unpacked-api")
    .add_attr_option<String>("interface-api")
    .set_default_keys({"cpu"});
```

This can be split into several smaller registries which better reflect what is being interacted with:
```
TVM_REGISTER_TARGET_KIND("c", kDLCPU)
    .add_attr_option<String>("mcpu")
    .add_attr_option<String>("march")
    .set_default_keys({"cpu"});

TVM_REGISTER_EXECUTOR_KIND("vm");
TVM_REGISTER_EXECUTOR_KIND("graph");
TVM_REGISTER_EXECUTOR_KIND("aot")
    .add_attr_option<Integer>("workspace-byte-alignment")
    .add_attr_option<Bool>("unpacked-api")
    .add_attr_option<String>("interface-api");

TVM_REGISTER_RUNTIME_KIND("c")
  .add_attr_option<Bool>("system-lib")
  .add_attr_option<Bool>("link-params", Bool(false))
  .add_attr_option<String>("runtime");
```

And then those attributes can be added as attributes on the `IRModule` before the entering `relay.build`:
```py
func = ... # some function
ir_mod = tvm.IRModule.from_expr(func)
ir_mod.set_attr("Runtime", Runtime({
  "kind": "c",
  "link-params": False
})
ir_mod.set_attr("Executor", Executor({
  "kind": "aot",
  "unpacked-api": False
})
relay.build(ir_mod)
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Most of the work involved here is creating independent registries for each type and the related wiring of Python objects to expose them publically - for example, creating `Executor` objects and `ExecutorRegistry` from `AttrRegistry`. Alongside this, the `IRModule` in Python needs to expose the `set_attr` method to enable composing this before entry into `relay.build`.

`tvmc` would be ammended to parse the relevant arguments based on the command line composition mechanism defined and construct an `IRModule` for TVM to use.

# Drawbacks
[drawbacks]: #drawbacks

This is a paradigm breaking change which results in `IRModule`s entering `relay.build` having to have prepared the attributes ahead of time.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

# Prior art
[prior-art]: #prior-art

TVM is full of registries, for `TargetKind`, `PassConfig` and a variety of other things, these are reflected in Python as well - so this is all within TVMs normal operating procedures.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

Do we want to treat `Target` similarly as we do this, placing them in an attribute on the `IRModule`? They're being annotated for `Function`s now which makes sense to the author.

# Future possibilities
[future-possibilities]: #future-possibilities

Migrating to a lot of smaller registries means we'll have more granular groupings and they can be leveraged more easily by interfaces such as `tvmc` and the command line composition from registries.