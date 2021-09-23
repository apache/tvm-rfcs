- Feature Name: TVM Script Namespace
- Start Date: 2021-09-23
- RFC PR: [apache/tvm-rfcs#0036](https://github.com/apache/tvm-rfcs/pull/36)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary

[summary]: #summary

This is a RFC for the new namespace and user interface for TVM script.

- Use tvm.script as the `root` namespace for all TVM script related stuff
- Use `tvm.script.tir` for `TIR`, and idiomatically import it as `T`, like Keras is usually imported as K
- To be consistent with the names of their resulting types, use
  - `tvm.script.ir_module` for `IRModule`
  - `T.prim_func` for `tir.PrimFunc`

# Motivation

[motivation]: #motivation

TVMScript is a fancy tools to write TIR in Python syntax. However, it's not fully compatible with native Python tools (e.g., pylint) because it is not a runnable program in Python.

We have following pain points:

- No Python auto-completion support
- Usually conflicts with pylint
- APIs scatter in namespaces like tvm.script, tvm.tir, tvm.script.ty
- Somewhat non-trivial to understand at first glance what the decorator generates

With the new proposal, we are able to provide type stubs that provides users with TVM scripts that work well with linting and auto-completion.

# Guide-level explanation

[guide-level-explanation]: #guide-level-explanation

Let's see an example for the same TIR program but in different syntax.

## Existing Syntax

```Python
from tvm import tir
# ^ here tir is not related to the script but we need to import it. Otherwise there are many lint errors.
from tvm.script import ty
# ^ ty is a sub-module for only type system, which makes no sense to import it global

@tvm.script.tir
class Module:
  def func(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128], dtype="float32")  # pylint error: tir does not have attr match_buffer
    B = tir.match_buffer(b, [128], dtype="float32")
    with tir.block([128, tir.reduce_axis(0, 128)], "C") as [i, k]:
        B[i] += A[i, k]
```

## New Syntax

```Python
from tvm.script import tir as T
# ^ there is a broadly accepted precedence in doing this in the python community:
#   from keras import backend as K

@tvm.script.ir_module                                   # it generates an IRModule
class Module:
  @T.prim_func                                          # it generates a PrimFunc explicitly
  def func(a: T.handle, b: T.handle):                   # return type is not necessary for PrimFunc
    A = T.match_buffer(a, [128, 128], dtype="float32")  # no pylint errors for match_buffer
    B = T.match_buffer(b, [128], dtype="float32")
    with T.block([128, T.reduce_axis(0, 128)], "C") as [i, k]:
        B[i] += A[i, k]
```

# Reference-level explanation

[reference-level-explanation]: #reference-level-explanation

Pylint and Python auto-completion will look the symbols (function definitions) in the module (e.g. `tvm.script.tir`). Just write (or generate) a concrete function in the module, then can enable auto-completion and pass Pylint checks.

# Drawbacks

[drawbacks]: #drawbacks

Here are some existing works based on current TVM Script syntax. It need a huge refactor to migrate it to the new one.

# Future possibilities

[future-possibilities]: #future-possibilities

Possible to write relay functions in the same IRModule if possible.
