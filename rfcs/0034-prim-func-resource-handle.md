- Feature Name: Wiring up the PrimFunc resource_handle
- Start Date: 14-09-2021
- RFC PR: [apache/tvm-rfcs#34](https://github.com/apache/tvm-rfcs/pull/34)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)


# Summary
[summary]: #summary
This RFC aims to address the need to pass a `resource_handle` to a `PrimFunc` which currently exists as the final parameter to a backend packed function but isn't directly populated by executors. The scope is purely the wiring exercise to enable a `resource_handle` to be attached to a `PrimFunc` and pass through an executors code generation.

# Motivation
[motivation]: #motivation

Currently the `resource_handle` is unpopulated in the case of the code generated from an executor via a `PrimFunc`, this is largely due to the `PrimFunc` having no means of tracking the argument which exists outside of the `inputs` and `outputs`. The primary motivating factor is passing device structures around, specifically the case of [the C Device API](https://github.com/apache/tvm-rfcs/pull/31).

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

When creating a `PrimFunc`, you will be able to attach an additional property to it to use as the `resource_handle`:

```
PrimFunc::PrimFunc(Array<tir::Var> params, Stmt body, Type ret_type,
                   Map<tir::Var, Buffer> buffer_map, DictAttrs attrs, tir::Var resource_handle,
                   Span span);
```

This `resource_handle` can be used within the body of the `PrimFunc` to allow for any calls to external functions, device APIs or other such uses. When generating out these `PrimFunc`s they would have the `resource_handle` appended to the signature which can be populated via executor code generation:

```
int32_t my_operator(..., void* resource_handle);
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

A `tir::Var` is added to `PrimFunc` in `include/tvm/tir/function.h` which enables a `PrimFunc` to track and use the `resource_handle` parameter. This will be used by both unpacked and packed APIs to pass the resource down without packing into `TVMValue`, instead as a `void *`. 

When this is packed in the lowering phase, the `resource_handle` will be assumed to exist as the last argument after being provided by the executor code generation:

```cpp
return Call(
    dtype,
    tvm::tir::builtin::tvm_call_cpacked(),
    input0,
    input1,
    output0,
    resource_handle
)
```


The eventual `Call` returned in `lower_tvm_builtin.c` contains the `resource_handle` by removing this final argument:

```cpp
auto arg_count = op->args.size() - 1;
resource_handle = op->args[arg_count];

// ... packing using arg_count reduced by one

return Call(
    op->dtype,
    tvm::tir::builtin::tvm_call_cpacked(),
    {
        op->args[0],
        scope.stack_value,
        scope.stack_tcode,
        ConstInt32(arg_stack_begin),
        ConstInt32(arg_stack_begin + op->args.size() - 1),
        resource_handle
    }
);
```

# Drawbacks
[drawbacks]: #drawbacks

* This changes the intrinsic `call_packed` to assume the final argument is the `resource_handle`, making it incompatible with previous releases
* The lack of structure in the `call_packed` means it'll be 

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

* Introduce another intrinsic and matching `call_cpacked` which have the suffix `_with_resource_handle` or similar - this means each code generator would have to implement additional intrinsics to support `resource_handle`s fully.

# Prior art
[prior-art]: #prior-art
* Uses the existing `resource_handle` in the TVM code which isn't currently propagated

# Unresolved questions
[unresolved-questions]: #unresolved-questions

# Future possibilities
[future-possibilities]: #future-possibilities