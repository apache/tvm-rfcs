- Feature Name: name_mangling_ir_modules
- Start Date: 2022-06-29
- RFC PR: [apache/tvm-rfcs#84](https://github.com/apache/tvm-rfcs/pull/84)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)
- Github PR: [apache/tvm#12066](https://github.com/apache/tvm/pull/12066)

# Summary
[summary]: #summary

This RFC proposes a clean-up of the current name mangling strategy.

# Motivation
[motivation]: #motivation

One reason for this RFC is that currently, it is difficult to know at various points in the compiler whether a `name_hint` is final or has yet to be mangled.  Name mangling is performed in various places:

- `tvm::runtime::get_name_mangled` prefixes a name (usually module name) with a function name and is called from
    - the AOT executor to create the main function and set the "`global_symbol"` attribute. It is also used to obtain the main function name and run it. See `AOTExecutorCodegen::CreateMainFunc` and `AOTExecutorCodegen::Run`
    - During TE lowering in `TECompilerImpl::Lower`.
    - In the `source_module.cc` to perform codegen for the C runtime.
    - In the `NameMangleExtFuncs::Run`, name mangling is applied to all the module functions before AOT codegen.
- `tvm::relay::tec::GetUniqueName` is used to avoid conflicts between multiple variables/functions that have the same name.
    - in the `TECompilerImpl::LowerInternal` and `LowerShape`.

Additionally, multiple `GlobalVars` having the same name are created throughout the code. In `TECompilerImpl::LowerInternal`, they are reduced to single values.

This RFC aims to unify the creation of unique `GlobalVars` and refactor the current `GlobalVar` name mangling to be done through a single entity. 

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

The changes are internal to TVM. 
The code writer is expected to avoid as much as possible calling the constructor of `GlobalVar` and instead use a `GlobalVarSupply` to generate one:
- The `GlobalVarSupply` is constructed by passing a `NameSupply` that is used to generate unique strings for the `name_hint` member of `GlobalVars`.
- The `NameSupply` can be constructed by passing a `String prefix` (usually a module name) that can then be prepended  to the `String`s generated: `auto name_supply = NameSupply(mod_name);`

A `GlobalVarSupply` can be derived from an existing `IRModule` or an array of `IRModules`. 
When a pass generates new `GlobalVars`, a `GlobalVarSupply` can be created from the current `IRModule`. Then, `GlobalVarSupply::FreshGlobal`
can be used to guarantee uniqueness of new `GlobalVars`.
```
GlobalVarSupply var_supply = GlobalVarSupply(mod);
GlobalVar var = var_supply->FreshGlobal(gv_name);
mod-Add(var, func);  
```
When generating a `GlobalVarSupply` from an existing `IRModule`, the module name is used as a prefix for 
new `GlobalVars`. The module name is expected to be specified as an attribute to the IRModule. Otherwise,
a default module name is used.

The `GlobalVarSupply` contains two methods to provide a `GlobalVar`:

- `GlobalVar UniqueGlobalFor(String name, bool add_prefix)` performs a cache lookup and returns a `GlobalVar`. If a miss occurs, a new `GlobalVar` is created, inserted into the cache, and returned. The `add_prefix` boolean defaults to `true`. If it is `false`, then the `prefix_` field will not be added to the `GlobalVar`.
- `GlobalVar FreshGlobal(const String name, bool add_prefix)` guarantees to return a newly constructed `GlobalVar` that is guaranteed not to conflict by name with other `GlobalVars` generated by the same `GlobalVarSupply` object. 
The functionality of `add_prefix` is as described above.

The name mangling in case of conflict is currently performed by appending `_1_2_3..._n` to the `name` parameter. 
An improvement is to treat the integer suffix 
separately from the string prefix. 
The following text snippet provides the general idea of mangling improvement:
```
x = get_next_name("fun12", existing_names=["func12", "func13", "func4"])
assert x == "func14"
``` 
This can be useful to avoid the current confusion caused by having multiple names in form:
```
func_1
func_1_2
func_1_2_3
```

<br></br>
The `NameSupply` can be used when `GlobalVars` are not needed but uniqueness of strings is required. 
The methods exposed by the `NameSupply` are similar to the ones exposed by the `GlobalVarSupply`:
- `String FreshName(const String& name, bool add_prefix)` to retreive a unique `String`.
- `String ReserveName(const String& name, bool add_prefix)` to mark a name as in use with the `NameSupply`.
- `bool ContainsName(const String& name, bool add_prefix)` to check if a `String` is already in use.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

The `GlobalVarSupply` and `NameSupply` will be implemented as a class in TVM following the usual design, extending `Object` and `ObjectRef` and will be accessible through FFI.
The `GlobalVarSupply` will contain a `NameSupply` and a map of `String -> GlobalVar`. 
The `NameSupply` will contain an internal map of `String -> Int` used to provide unique names.

Additional refactorings to the compiler will be performed as part of this RFC. For example, the `GlobalVar` deduplication in `TECompilerImpl::LowerInternal` will be removed by changing the signature of `tvm::LowerSchedule`.

# Drawbacks
[drawbacks]: #drawbacks

- There might be cases when creating new `GlobalVars` without using a `GlobalVarSupply` might be needed. For example, in `IRModule::FromExprInContext`, the expression might already be annotated with a global symbol.
We must be sure that the `GlobalVarSupply` interacts well with the `global_symbol` attribute. This can be done by calling 
`GlobalVarSupply::UniqueGlobalFor(global_symbol')`.
- Ensuring that the `GlobalVarSupply` and `NameSupply` are always used when possible is not enforceable. This RFC aims to
deduplicate the various implementations of `GetUniqueName`. 

# **Rationale and alternatives**
An alternative way to unify name supplying and mangling in IR modules is to create a compiler pass that does the job. 
The benefit of this is that the implementation would not be very intrusive. The downside is that it could only address name mangling inside IRModules. 
There are cases when String uniqueness is required outside an IRModule. 

# Future possibilities
[future-possibilities]: #future-possibilities

Currently, the name mangling method is hardcoded in `NameSupply`. It might prove beneficial in the future
to allow a lambda function to be passed when constructing a `NameSupply`. 

