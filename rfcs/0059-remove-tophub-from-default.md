- Feature Name: remove_tophub_from_default
- Start Date: 2021-03-04
- RFC PR: [apache/tvm-rfcs#0059](https://github.com/apache/tvm-rfcs/pull/0059)
- GitHub Issue: [apache/tvm#10474](https://github.com/apache/tvm/pull/10474)

# Summary
[summary]: #summary

Remove tophub from being the default autotvm configuration when compiling with
the VM or graph executor. This bring thes VM and graph executor compilation
workflow in line with all other workflows which do no use tophub by default.

https://github.com/apache/tvm/pull/10474 provides a short discussion of the
tradeoffs/issues in applying this change.

# Motivation
[motivation]: #motivation

- Tophub is not maintained and is out of date in some cases. Users should
  explicitly use it so there is no surprising behavior in using out of date
  schedules.
- Tophub is implicitly used only with the VM and graph executor compilation
  flows. Other flows (like `VMCompiler.optimize`) will return different
  results because they do not implicitly use tophub.
- Many default schedules exist in TVM, but they are not used because the
  tophub schedules override them. Either they should be removed or used.
- Compilation implicitly depends on scheduling choices located outside of the
  TVM codebase. This implicit behavior is confusing when debugging and
  requires us to maintain a consistent state between two different repos.

Users can always use the tophub context explicitly like so:

```python
with autotvm.tophub.context(target):
  ...
```

# Drawbacks
[drawbacks]: #drawbacks

We may see a performance degradation in models compiled without tuning as the
tophub schedules may be more optimized then the fallback/untuned schedules. If
this is an issue, we can port the poorly performing schedules from tophub to
the fallback (i.e. default) configuration for the op.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

Alternative choices:

- Use tophub as the default context when no other context has been specified.
  This will make `VMCompiler.optimize` and `VMCompiler.lower` consistent in
  their results.

- Add tophub to `VMCompiler.optimize`. This doesn't solve the underlying
  problem. I expect us see more inconsistent results in other parts of the
  codebase where tophub is not used.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

Will there be any actual performance impact in removing tophub from being the default?

Should we provide a deprecation notice, or a change notice, or no notice at all?
