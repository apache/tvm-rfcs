- Feature Name: Update TVM Script block syntax
- Start Date: 2021-10-06
- RFC PR: [apache/tvm-rfcs#0041](https://github.com/apache/tvm-rfcs/pull/41)

# Summary

[summary]: #summary

This is a RFC for the new syntax for blocks in TVM Script:

- Disable auto-complete nesting loops
- Use `T.axis.S` and `T.axis.R` for block var defining and value binding.
- Use `T.axis.remap` for trivial bindings.

# Motivation

[motivation]: #motivation

Block is the core data structure in TensorIR, meanwhile, TVMScript is one of the major input to TensorIR. Current block syntax in TVMScript does a good job but still can be better.

We have following pain points:

## Lines can be very long if a block has many block var

```Python
# An example block for conv2d on NHWCnc (packed layout for TensorCore)
with tir.block([2, 14, 14, 4, tir.reduce_axis(0, 2), tir.reduce_axis(0, 3),
                tir.reduce_axis(0, 3), 16, 16, tir.reduce_axis(0, 16)], "Conv") as \
        [n, h, w, o, ic, kh, kw, nn, oo, ii]:
    with tir.init():
        C[n, h, w, o, nn, oo] = tir.float32(0)
    C[n, h, w, o, nn, oo] = C[n, h, w, o, nn, oo] \
                            + tir.cast(Apad[n, h + kh, w + kw, ic, nn, ii], "float32") \
                            * tir.cast(W[kh, kw, ic, o, ii, oo], "float32")
```

## Unreasonable loop completion

In order to make TVMScript easy to write, we enable auto-completion to blocks. Currently, we have two loop completion rules:

- Auto map trivial values: if the number of block vars is equal to the number of nested loops, bind them.

  ```Python
  for i, j in T.grid(16, 16):
      with T.block([16, 16]) as [vi, vj]:
          # T.bind(i, vi)    <- auto-completion
          # T.bind(j, vj)    <- auto-completion
          ...
  ```

- Auto generate nested loops: generate loop nesting and bind them if there is no loop out of block.

  ```Python
  # for i, j in T.grid(16, 16):  <- auto-completion
      with T.block([16, 16]) as [vi, vj]:
          # T.bind(i, vi)        <- auto-completion
          # T.bind(j, vj)        <- auto-completion
          ...
  ```

Both rules are too *SMART*, which may confuse the users.

# Guide-level explanation

[guide-level-explanation]: #guide-level-explanation

Based on those two pain points, we design a new block syntax for TensorIR, which no longer has too *SMART* completion and too long lines but also easy to write.

## Complete Form

```Python
for i, j, k in T.grid(512, 512, 512):
    with T.block("name"):
        vi = T.axis.spatial((0, 512), i)
        # (0, 512) for the block var iter_dom, can be write as 512 if starts from 0
        vj = T.axis.spatial(512, j)
        # vj = T.axis.S(512, j)   <- we can use `S` for spatial.
        vk = T.axis.reduce(512, k)
        # vk = T.axis.R(512, k)   <- we can use `R` for reduce.
        T.reads(...)            # <- access region still can be detected.
        T.writes(...)
        ...
```

## A sugar for trivial bindings

```Python
for i, j, k in T.grid(512, 512, 512):
    with T.block("name"):
        # SSR means [spatial, spatial, reduce] for three vars
        # Only trivial bindings are allowed here since we need to detect iter_dom from the loops
        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
        ...
```

## A Mixed format

```Python
for io, ii, j, k in T.grid(16, 32, 512, 512):
    with T.block("name"):
        vi = T.axis.S(512, io * 32 + ii)
        vj, vk = T.axis.remap("SR", [j, k])
        ...
```

# Reference-level explanation

[reference-level-explanation]: #reference-level-explanation

It's almost an user interface change, so might not have many technical explanations. Only one thing notable: the block var is a ordered list rather than a list. See an example:

```Python
for i, jo, ji, k in T.grid(512, 32, 16, 512):
    with T.block("A"):
        vi = T.axis.S(512, i)
        vj = T.axis.S(512, jo * 32 + ji)
        vk = T.axis.R(512, k)
        ...

for i, jo, ji, k in T.grid(512, 32, 16, 512):
    with T.block("B"):
        vi, vk = T.axis.remap("SR", [i, k])
        vj = T.axis.S(512, jo * 32 + ji)
        ...
```

`block A` (block vars:`[vi, vj, vk]`) is different from `block B` (block vars:`[vi, vk, vj]`)


# Drawbacks

[drawbacks]: #drawbacks

- Here are some existing works based on current TVM Script syntax. It need some refactor to migrate it to the new one.

- Some early developers get used to the old format, may bring some extra effort to move to the new one.

# Future possibilities

[future-possibilities]: #future-possibilities

Iter domain may be detected from any PrimExpr which is affine.
