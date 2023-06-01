# TIR Language Specification (draft)

TensorIR (TIR) is TVM's intermediate representation (IR) for describing operations over tensors. In particular, TIR allows for expressing low-level operations on tensors in a manner that is amenable to automatically implementing common optimization strategies like loop tiling or unrolling, especially in the context of autotuning, wherein these transformations are applied using a learned cost function.

This document is intended to serve primarily as a high-level reference for TIR's semantics (observable behavior), aiming to describe a high-level, portable subset of TIR as it is meant to be initially passed to the compiler. The subset of TIR described in this document should be accepted by the compiler and lowered to any hardware back-end without issue; a reader of this document should be able to correctly describe how each language construct will ultimately be executed and give the final result of running the program. Thus, this specification can serve as a reference for users of TIR's front-end (who can ensure their programs will behave as intended) as well as compiler implementers (who can ensure different compiler optimizations maintain the guarantees on visible behavior provided by the specification).

There are two main reasons this document describes only a subset of TIR. The first is that, unlike many programming languages, the grammar includes lots of auxiliary information intended for the compiler's internal use. Skillful users of TIR can take advantage of the compiler implementation by specifying this additional information appropriately, but the compiler implementation is greatly subject to change and it is unclear that such behavior should be part of the contract between users and the compiler. Indeed, one purpose of this specification is to clarify the distinction between the "front-end" interface to TIR and the compiler internals, since this has previously been ambiguous due to the degree to which the TIR implementation exposes compiler internals. The second reason is that the hardware back-ends supported by TIR have greatly varying properties. While it is possible that some of these details might be specified in the future or described in separate documents, this version of the specification will not account for TIR programs that make use of certain internal details of the compiler implementation or low-level details of specific hardware back-ends. The aim, rather, is to establish simple ground rules for the language that account for its most common uses.

_Note: This specification corresponds to the intended functionality of the language, which may differ from how it has been implemented. Portions of the specification that differ from the implementation will be given in «double caret marks (guillemets)» (color-coding would be preferable, but Github Markdown does not support it). These discrepancies should be corrected or addressed._

## Overview

TIR is an imperative language that describes tensor operations mainly in terms of bounded loops over indices with defined iteration domains, wherein the loop bodies apply scalar operations to tensor elements. The main abstractions in TIR are values (scalars or vectors) and buffers (regions of memory, which tend to represent tensors); elements from buffers can be read and written in loop bodies in order to implement operations on tensors. In most cases, the bounded ranges for loop iterations provide the compiler with more information for optimizations and autotuning procedures.

In addition to its primary functionality in describing loops over tensor elements, TIR is also capable of interfacing with TVM's object system and can invoke arbitrary packed functions via intrinsics, though this version of the specification will not go into detail on intrinsics.

As noted in the preamble, this version of the specification covers a portable, high-level subset of TIR, excluding in particular behaviors that might make use of compiler implementation internal details or low-level properties of hardware back-ends. While such uses of TIR exist and will continue to exist, we defer specifying them until future versions of this specification in order to establish basic expectations for the language's behavior and additionally in order to avoid "committing" the compiler implementers to supporting certain behaviors unto perpetuity; these lower-level details generally reflect conditions of the deep learning stack that are highly subject to change. Instead, in this version of the specification, we account for high-level uses of TIR intended to correspond to very common applications:

* The output of the TE (Tensor Expressions) library,
* Implementations of new tensor operators intended to be used with TVM's auto-tuning libraries, and
* TIR code intended to be invoked from Relax via TVM Unity.

This document will note which features are outside the subset of TIR intended to be specified at present; any program that makes use of these features or does not abide by the restrictions described is thus considered to be unspecified: the specification makes no guarantees on that program's behavior.

## Grammar

Notation: `[x]` means a sequence (zero or more) of `x`, `{x: y}` means "a map from `x` to `y`," and `x?` means "optionally `x`."

```python
PrimFunc ::= PrimFunc(params: [Var], body: Stmt, ret_type: Type?, 
                      buffer_map: {Var: Buffer}, attrs: Attrs)

Type ::=
    PrimType(dtype: DataType)
  | PointerType(element_type: Type, storage_scope: str)
  | TupleType(fields: [Type])

DataType ::= DataType(code: DTTypeCode, bits: DTSize, lanes: DTLanes)

DTTypeCode ::= Int() | UInt() | Float() | BFloat() | Handle()
DTSize ::= 0 | 1 | 8 | 16 | 32 | 64
DTLanes ::= 1 | 4 | 8 | 16 | 32 | 64

Stmt ::=
    LetStmt(var: Var, value: PrimExpr, body: Stmt)
  | AttrStmt(node: ObjectRef**, attr_key: str, value: PrimExpr, body: Stmt)
  | AssertStmt(condition: PrimExpr, message: PrimExpr, body: Stmt)
  | BufferStore(buffer: Buffer, value: PrimExpr, indices: [PrimExpr])
  | BufferRealize(buffer: Buffer, bounds: [Range], condition: PrimExpr, body: Stmt)
  | Allocate(buffer_var: Var, dtype: DataType, extents: [PrimExpr], 
             condition: PrimExpr, body: Stmt, annotations: {str: Object*})
  | DeclBuffer(buffer: Buffer, body: Stmt)
  | SeqStmt(seq: [Stmt])
  | IfThenElse(condition: PrimExpr, then_case: Stmt, else_case: Stmt?)
  | Evaluate(value: PrimExpr)
  | For(loop_var: Var, min: PrimExpr, extent: PrimExpr, 
        kind: ForKind, body: Stmt, thread_binding: IterVar?, annotations: {str: Object*})
  | While(condition: PrimExpr, body: Stmt)
  | Block(iter_vars: [IterVar], reads: [BufferRegion], 
          writes: [BufferRegion], name_hint: str, body: Stmt,
          init: Stmt?, alloc_buffers: [Buffer], 
          match_buffers: [MatchBufferRegion],
          annotations: {str: Object*})
  | BlockRealize(values: [PrimExpr], predicate: PrimExpr, block: Block)

Buffer ::= Buffer(data: Var, dtype: DataType, shape: [PrimExpr], 
                  axis_separators: [IntImm], strides: [PrimExpr], 
                  elem_offset: PrimExpr?, name: str,  data_alignment: int, 
                  offset_factor: int, buffer_Type: BufferType)

BufferType ::=
     kDefault()
   | kAutoBroadcast()

BufferRegion ::= BufferRegion(buffer: Buffer, region: [Range])

MatchBufferRegion ::= MatchBufferRegion(buffer: Buffer, source: BufferRegion)

PrimExpr ::=
    Var(name_hint: str, dtype: DataType, type_annotation: Type)
  | IntImm(value: int, dtype: DataType)
  | FloatImm(value: float, dtype: DataType)
  | StringImm(value: str)
  | Cast(value: PrimExpr, dtype: DataType)
  | Select(condition: PrimExpr, true_value: PrimExpr, false_value: PrimExpr)
  | BufferLoad(buffer: Buffer, indices: [PrimExpr])
  | Ramp(base: PrimExpr, stride: PrimExpr, lanes: int)
  | Broadcast(value: PrimExpr, lanes: int)
  | Let(var: Var, value: PrimExpr, body: PrimExpr)
  | Call(dtype: DataType, op: Op|GlobalVar, args: [PrimExpr])
  | Shuffle(vectors: [PrimExpr], indices: [PrimExpr])
  | BinaryOp
  | CmpOp
  | LogicalOp

LogicalOp ::=
    And(a: PrimExpr, b: PrimExpr)
  | Or(a: PrimExpr, b: PrimExpr)
  | Not(a: PrimExpr)

BinaryOp ::=
    Add(a: PrimExpr, b: PrimExpr)
  | Sub(a: PrimExpr, b: PrimExpr)
  | Mul(a: PrimExpr, b: PrimExpr)
  | Div(a: PrimExpr, b: PrimExpr)
  | Mod(a: PrimExpr, b: PrimExpr)
  | FloorDiv(a: PrimExpr, b: PrimExpr)
  | FloorMod(a: PrimExpr, b: PrimExpr)
  | Min(a: PrimExpr, b: PrimExpr)
  | Max(a: PrimExpr, b: PrimExpr)

CmpOp ::=
    Eq(a: PrimExpr, b: PrimExpr)
  | NE(a: PrimExpr, b: PrimExpr)
  | LT(a: PrimExpr, b: PrimExpr)
  | LE(a: PrimExpr, b: PrimExpr)
  | GE(a: PrimExpr, b: PrimExpr)
  | GT(a: PrimExpr, b: PrimExpr)

ForKind ::=
    kSerial()
  | kParallel()
  | kVectorized()
  | kUnrolled()
  | kThreadBinding()

Range ::= Range(min: PrimExpr, extent: PrimExpr?)

IterVar ::= IterVar(dom: Range?, var: Var, iter_type: IterVarType, thread_tag: str)

IterVarType ::=
     kDataPar()
   | kThreadIndex()
   | kCommReduce()
   | kOrdered()
   | kOpaque()
   | kUnrolled()
   | kVectorized()
   | kParallelized()
   | kTensorized()

Attrs ::= Attrs(contents: {str: Object*})
```

*Note that attributes and annotations can contain arbitrary TVM objects as values. These objects are used only at compile time.

**Refers to one of the base classes in the TVM object representation. In practice, `ObjectRef`s are usually TIR AST nodes. Which ones are appropriate depend on the specific attributes (only those listed under the semantics have any visible effects; the rest are used only at compile time).

Additionally, at run time, `PrimFunc`s take in parameters corresponding to buffers via the `DLPack` library's `DLTensor` class, defined in [dlpack.h](https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h):
```C
typedef struct {
  void* data; // pointer to the buffer contents
  DLDevice device; // not discussed in this specification
  int32_t ndim;
  DLDataType dtype; // has the same fields as DataType in the above AST
  int64_t* shape; // array giving the shape of the corresponding buffer
  int64_t* strides; // can be null
  uint64_t byte_offset;
} DLTensor;
```

The correspondence of the fields of `DLTensor` to `Buffer` will be discussed with the semantics for invoking a `PrimFunc`.

## Values

Expressions in TIR (PrimExprs) operate on three kinds of values:
1. Scalars, which are single members of TIR's numerical datatypes: Floating point (`Float`), Brain floating point (`BFloat`), signed integer (`Int`), unsigned integer (`UInt`). `Int` and `UInt` values can have bitwidths of 8, 16, 32, or 64. `UInt` values can also have a bitwidth of 1 (corresponding to a Boolean value). Scalar values are immutable. `Float` values can have bitwidths of 16, 32, or 64. `BFloat` values must have a bitwidth of 16.
2. Vectors, which correspond to an immutable grouping of multiple members of the above-mentioned datatypes: `Float`, `BFloat`, `Int`, or `UInt`, with the same bitwidths permitted for `Float`, `BFloat`, `Int`, and `UInt` scalars. Vectors may contain 4, 8, 16, 32, or 64 elements of the listed data type. Their representation at run time is back-end–specific, so we make no stipulations about how the data in a vector is represented.
3. Pointer values (which have the `Handle` datatype, indicating that they are "handles" to data), which are indices to memory locations that contain scalars or other data of interest. Pointers may address a value with a known datatype (`Float`, `BFloat`, `Int`, or `UInt`), or they may address values whose datatype is unknown at compile time or opaque data intended only for calls to builtins (external procedures). In principle, a pointer could be used to address a vector, but this is presently not supported. To avoid confusion with `PointerType` below, we will generally refer to pointer values as "handles" in this document.

Note that a pointer is simply an index into memory; the management of the memory is part of the program state. In TIR, regions of memory that are valid to address via pointers are commonly indicated in the AST using the `Buffer` construct, which defines the size and arrangement of data in some region and defines other information, such as the buffer's shape and stride size. However, "buffers" themselves in TIR are not values in the language: they are not returned by expressions and manifest at run time only as handles.

## Notation Used for Discussing ASTs

For convenience in the specification, we will use some shorthand for common concepts:

* `node->field`: This refers to the field named `field` on an AST node named `node`.
* `list[index]`: This refers to the `index`th element of a list called `list`, using zero-based indexing.
* `vector.index`: This refers to the `index`th element of a vector value called `vector`, using zero-based indexing. This notation is meant to distinguish vector values in TIR from lists.
* `len(list)`: This gives the length of `list`.
* `||vector||`: This gives the length of `vector`. (Again, this notation is meant to distinguish vector values from lists.)
* `dtype(expr)`: This will be used to denote the datatype derived from a given expression `expr`. The section below will describe how datatypes are derived. This "function" should be distinct from AST fields named `dtype` (e.g., for `Buffer` nodes).

## Types

### Datatypes

All TIR `PrimExpr`s have an associated `DataType` that describes the datatype of the result of evaluating the `PrimExpr`. These are defined in [`include/tvm/runtime/data_type.h`](https://github.com/apache/tvm/blob/main/include/tvm/runtime/data_type.h), as `PrimExprNode::dtype`.

DataTypes have three fields:
1. `code`, which describes the type of elements in the datatype. The following are the type codes used in TIR (note: this document will leave off the initial `k` when referring to these codes for readability):
    * `kInt` and `kUInt` for signed and unsigned integer values, respectively
    * `kFloat` for floating point values and `kBFloat` for the `bfloat16` format (Brain floating point).
    * `kHandle` for pointer values (handles).
2. `bits`, which describes the bitwidth of the elements of the datatype. Common bitwidths in TIR are 1, 8, 16, 32, and 64 (for integers)
3. `lanes`, which describes the number of elements in a vector value. lanes is 1 for a scalar value and greater than 1 for a vector (4, 8, 16, 32, or 64 lanes are common in TIR).

### Definitions Related to DataType

* **Scalar**. A DataType with the `Int`, `UInt`, `Float`, or `BFloat` codes is a scalar type if `lanes` is 1.
* **Vector**. A DataType with the `Int`, `UInt`, `Float`, or `BFloat` codes is a vector type if `lanes` is greater than 1.
* **Handle (Pointer)**. A DataType is a handle type if it has the `Handle` code. The value of `bits` (if nonzero) corresponds to the size of the pointer (this is 64 on most devices supported by TIR, but pointers on some lower-powered devices are 32 bits wide). (As aforementioned, we use the term "handle" to distinguish from `PointerType` in the below section.) The `lanes` field is undefined for handle values, though the implementation always sets it to 1. Note that if `bits` is 0, then it instead refers to the `Void` type.
* **Boolean (`Bool`)**. Refers to a `UInt` datatype with a bitwidth of 1, since these are used to represent the results of logical operators in TIR. `Bool` datatypes can be either scalars (1 lane) or vectors (4, 8, 16, 32, or 64 lanes).
* **`Void`**. Refers to a `Handle` datatype with a bitwidth of 0, indicating an opaque object inaccessible to TIR (but which may be used by calls to builtins).

### Notation

These notations are used only sparingly in this specification, but are often used in TIR's documention:
* For scalars and pointers: The string format is `{code}{bits}`, with the code in lowercase. For example, `int32` refers to 32-bit integers and `float16` refers to 16-bit floating-point numbers.
* For rectors: The string format is `{code}{bits}x{lanes}`. For example, a vector of 16-bit floating point values with 4 members is `float16x4` and and a vector of 8-bit integers with 8 members is `int8x8`.

### Finer-Grained Types

TIR `PrimExpr`s have a datatype (accessible in the implementation via the `dtype` field) that indicates the datatype resulting from evaluating the expression. However, TIR variables can have a finer-grained type in their `type_annotation` field. These finer-grained types are denoted in the AST under `Type` (`tvm::ir::Type` in the implementation).

The finer-grained types are as follows:
1. `PrimType` indicates that the `Var` does not have a more refined type than its `DataType` and provides no further information. It is required that the `dtype` field of the `PrimType` be equal to the `dtype` field of the `Var`.
2. `PointerType` indicates that the Var is bound to a pointer value. `PointerType`, unlike the `Handle` datatype, describes the datatype of the value being referenced by the pointer. This is most often used for the `data` field of a `Buffer`, as the `data` field is a pointer to a specific region of memory on a specific device. If a `Var` has a `type_annotation` that is a `PointerType`, its `dtype` field must have the `kHandle` code. There are two fields in `PointerType`:
    * `element_type`: A `Type` (which must be `PrimType`) that describes the type of the value the pointer refers to.
    * `storage_scope`: A string that conveys device-specific information regarding the region of memory that the pointer addresses. For example, "`shared`" refers to shared memory, and "`shared.dyn`" refers to dynamic shared memory on CUDA GPUs.
3. `TupleType` is used much less frequently than the previous two. Namely, it is used in only two cases in TIR, namely for the `ret_type` field of `PrimFunc`s or as the `type_annotation` field for a `Var` that references a value with a `Void` datatype. In these cases, the `TupleType` must have an empty list for its `fields` value (in the case of a `PrimFunc`, it means that it does not return a value; for a `Var`, it means that the value is `Void`).

### Rules for Assigning Datatypes to PrimExprs

For each `PrimExpr`, we define the rule determining their `dtype` field below:

1. `Var(name_hint, dtype, type_annotation)`: There are two ways to construct a Var, either by specifying dtype or type_annotation (and not the other):
    1. If `dtype` is specified and `type_annotation` is not specified, then the resulting datatype is `dtype`. If dtype is not `Void`, then `type_annotation` should be set to `PrimType(dtype)`. If dtype is `Void`, then `type_annotation` should be set to `TupleType([])`.
    2. If `type_annotation` is specified and `dtype` is not, then determine the resulting datatype based on `type_annotation` as follows:
        1. If `type_annotation` is `PrimType`, then the resulting datatype is `type_annotation->dtype`.
        2. If `type_annotation` is `PointerType`, then the resulting datatype is `DataType(Handle, 64, 1)`.
        3. If type_annotation is `TupleType([])`, then the resulting datatype is `Void`.
        4. Any other `type_annotation` should be considered invalid and result in a type error.
2. `IntImm(value, dtype)`:
    1. The following conditions must hold or else there is a type error:
        1. `dtype` must be a scalar (have exactly one lane).
        2. `dtype` must have a typecode of either `Int` or `UInt`.
        3. If `dtype->code` is `UInt`, `value` must be greater than or equal to 0. If `dtype->bits` is less than 64, value must be strictly less than $2^b$, where $b$ is `dtype->bits`.
        4. If the `dtype->code` is `Int` and `dtype->bits` is greater than 1 and less than 64, value must be greater than or equal to $-(2^{b-1})$ and strictly less than $2^{b-1}$, where $b$ is `dtype->bits`. If the bitwidth is exactly 1, then `value` must be either 0 or 1.
    2. The resulting datatype is `dtype`.
3. FloatImm(value, dtype):
    1. The following conditions must hold or else there is a type error:
        1. `dtype->code` must be `Float` or `BFloat`
        2. `value` must be `NaN`, `+inf`, `-inf`, or between the minimum and maximum values for a floating point number of the bitwidth given: for 16-bit `Float`s: $\pm 65504$; for 16-bit `BFloat`s: $\pm 3.38953139 \cdot 10^{38}$; for 32 bits: $\pm 3.402823466 \cdot 10^{38}$; and for 64 bits: $\pm 1.7976931348623158 \cdot 10^{308}$.
    2. The resulting datatype is `dtype`.
4. `StringImm(value)`: Its datatype is `DataType(Handle, 64, 1)`.
5. `Cast(value, dtype)`: The number of lanes in `dtype(value)` must match the number of lanes in `dtype` or else there is a type error. «If `value` has a `Handle` datatype, then `dtype` must also be `Handle` or else there is a type error; if `dtype` is `Handle`, then `value` must have a typecode of `Int`, `UInt`, or `Handle` or else there is a type error.» The resulting datatype is `dtype`.
6. `Select(condition, true_value, false_value)`:
    1. The following conditions must hold, or else there is a type error:
        1. `dtype(condition)` must be a `Bool` datatype (not necessarily a scalar).
        2. `dtype(true_value)` and `dtype(false_value)` must match.
        3. `dtype(condition)->lanes` must either be 1 or match `dtype(true_value)->lanes`.
    2. The resulting datatype is `dtype(true_value)`.
7. `BufferLoad(buffer, indices)`:
    1. Suppose `len(indices)` is `n`. If `n` is greater than 0, the first `n - 1` members of indices must have a scalar datatype (i.e., exactly one lane). That is, all indices except the last one must have scalar data types.
    2. «All members of `indices` must have datatypes with `Int` or `UInt` typecodes, and they must all have the same bitwidth. In principle, hardware back-ends have some specific size of index they expect (most commonly 64-bit, but it may be 32-bit or 16-bit on lower-powered systems), but any integer width is permitted in TIR (it will be cast to the expected width at run time).»
    3. Let `index_lanes` be 1 if `indices` is of length 0. If `len(indices) > 0`, then let `index_lanes` be `dtype(indices[len(indices)-1])->lanes` (the last member's `lanes`).
    4. Let `buffer_lanes` be `buffer->dtype->lanes`.
    5. The resulting datatype will be `DataType(code=buffer->code, bits=buffer->bits, lanes=index_lanes*buffer_lanes)`.
8. `Ramp(base, stride, lanes)`:
    1. The following conditions must hold or else there is a type error:
        1. The value of `lanes` must be strictly greater than 1.
        2. `dtype(base)` and `dtype(stride)` must match and `dtype(base)->lanes` and `dtype(stride)->lanes` must both be 1.
        3. «`dtype(base)->code` and `dtype(stride)->code` must be `Int` or `UInt`.»
    2. The resulting datatype will be `DataType(code=dtype(base)->code, base=dtype(base)->bits, lanes=lanes)`.
9. `Broadcast(value, lanes)`:
    1. The following conditions must hold or else there is a type error:
        1. `dtype(value)->lanes` must be 1.
        2. `lanes` must be strictly greater than 1.
    2. The resulting datatype will be `DataType(code=dtype(value)->code, bits=dtype(value)->bits, lanes=lanes)`.
10. `Let(var, value, body)`:
    1. `dtype(var)` must match `dtype(value)` or else there is a type error.
    2. The resulting datatype is `dtype(body)`.
11. `Call(dtype, op, args)`: The resulting datatype is `dtype`; the datatype is not otherwise checked.
12. `Shuffle(vectors, indices)`:
    1. «`len(vectors)` must be at least 1 or else there is a type error.»
    2. The datatypes of all elements of `vectors` must have the same typecode and bitwidth; it is a type error otherwise. Let the typecode be `vector_code` and the bitwidth be `vector_bits`.
    3. Let `total_lanes` be the sum of `dtype(vectors[i])->lanes` over all `i` from 0 to `len(vectors) - 1`, inclusive.
    4. `len(indices)` must equal `total_lanes` or else it is a type error.
    5. «All members of `indices` must be `Int` or `UInt` scalars or else it is a type error.»
    6. The resulting datatype will be `DataType(code=vector_code, bits=vector_bits, lanes=total_lanes).`
13. Binary ops (with arguments `a` and `b`), which are `Add`, `Sub`, `Mul`, `Div`, `Mod`, `FloorDiv`, `FloorMod`, `Min`, and `Max`: `dtype(a)` and `dtype(b)` must match «and the typecode must not be `Handle`», or else it is a type error. The result will be `dtype(a)`. «For `Mod`, `dtype(a)` and `dtype(b)` must also both have either the `Int` or `UInt` typecode.»
14. Logical ops `And` and `Or`, with arguments `a` and `b`: `a` and `b` must both have `Bool` datatypes and have the same number of lanes. The result will be `dtype(a)`.
15. `Not(a)`: `a` must have a `Bool` datatype (or else it is a type error). The result will have a `Bool` datatype with the same number of lanes as `dtype(a)`.
16. Comparison operators (with arguments `a` and `b`), which are `Eq`, `NE`, `LT`, `LE`, `GE`, and `GT`: `a` and `b` must have the same datatype «and the typecode must not be `Handle`», or else it is a type error. The result has a `Bool` datatype and the same number of lanes as `dtype(a)`.

### Typing Rules for Statements

Even though statements do not produce values themselves, many contain `PrimExpr`s and have requirements on the types for those `PrimExpr`s. There is a type error if any condition listed below does not hold for the given statement (assuming its subfields typecheck individually). Some of these rules also include structural requirements not related to datatypes or type annotations.

1. `LetStmt(var, value, body)`: If `var->type_annotation` is a `PointerType`, then `dtype(value)->code` must be `Handle` and `dtype(value)->bits` must be nonzero. (Note that there is no requirement on the `element_type` field: This allows for implicit casts of pointers.) Otherwise, `dtype(var)` and `dtype(value)` must match.
2. `AttrStmt(node, attr_key, value, body)`: Always valid.
3. `AssertStmt(condition, message, body)`: `message` must be either a `PrimExpr` whose datatype is `int32` or a `StringImm` node. Additionally, `condition` must be a `Bool` scalar.
4. `BufferStore(buffer, value, indices)`:
    1. Suppose `len(indices)` is `n`. If `n` is greater than 0, the first `n - 1` members of indices must have a scalar datatype (i.e., exactly one lane). That is, all indices except the last one must have scalar data types.
    2. Let `index_lanes` be 1 if `len(indices)` is 0. If `len(indices)` > 0, then let `index_lanes` be `dtype(indices[len(indices)-1])->lanes` (the lanes of the last member's datatype).
    3. «All members of `indices` must have a typecode of `Int` or `UInt`; in fact, they must all have the same typecode and bitwidth.»
    4. Let `buffer_lanes` be `buffer->dtype->lanes`.
    5. `dtype(value)->lanes` must be equal to `index_lanes * buffer_lanes`.
5. `BufferRealize(buffer, bounds, condition, body)`: The following conditions must hold:
    1. «`condition` is a `Bool` scalar.»
    2. «All members of bounds are `Int` or `UInt` scalars; their datatypes and bitwidths match.»
6. `Allocate(buffer_var, dtype, extents, condition, body, annotations)`: The following conditions must hold:
    1. Either `buffer_var->type_annotation` is `PointerType(dtype)` or `dtype` is a `Bool` scalar and `buffer_var->type_annotation` is `PointerType(int8)`.
    2. All members of extents must have a scalar datatype. «All members' typecodes and bitwidths must be Int or UInt and must all match.»
    3. `dtype(condition)` must be a `Bool` scalar.
7. `DeclBuffer(buffer, body)`: Always valid.
8. `SeqStmt(seq)`: Always valid.
9. `IfThenElse(condition, then_case, else_case)`: `dtype(condition)` must be a `Bool` scalar.
10. `Evaluate(value)`: Always valid.
11. `For(loop_var, min, extent, kind, body, thread_binding, annotations)`: The following conditions must hold:
    1. The datatypes of `min`, `extent`, and `loop_var` must all be scalars «with an `Int` or `UInt` typecode».
    2. `dtype(min)->bits` and `dtype(extent)->bits` must be less than or equal to `dtype(loop_var)->bits`.
    3. If `min` is an `IntImm` node and `dtype(min)->bits` < `dtype(loop_var)->bits`, then consider `min` to have the same datatype as `loop_var` (i.e., "promote" its datatype).
    4. If `extent` is an `IntImm` node and `dtype(extent)->bits < dtype(loop_var)->bits`, then "promote" its datatype as with `min`.
    5. After performing the datatype "promotions," if necessary, `dtype(loop_var)`, `dtype(min)`, and `dtype(extent)` must all match exactly.
    6. «If `kind` is `kVectorized`, `body` must not contain `While` statements. Additionally, `min` must be an `IntImm` with a value of 0 and extent must be an `IntImm` with a value of at least 1.»
12. `While(condition, body)`: `condition` must have a scalar datatype with an `Int` or `UInt` typecode. Additionally, `condition` must not be an `IntImm` node.
13. `Block(iter_vars, reads, writes, name_hint, body, init, alloc_buffers, match_buffers, annotations)`: «The datatypes of the `var` fields for all members of `iter_vars` must be `Int` or `UInt` scalars.»
14. `BlockRealize(iter_values, predicate, block)`: `len(iter_values)` must match `len(block->iter_vars)`. Additionally, `predicate` must have a `Bool` datatype.

### Typing Rules for Other Language Constructs

Certain language constructs like `IterVar` are neither `PrimExpr`s nor statements but are used to construct `PrimExpr`s and statements. They have some typing rules as well. The conditions listed below for each construct must hold (assuming their subfields type check) or else there is a type error.

1. `IterVar(dom, var, iter_type, thread_tag)`: If `dom` is specified and the `dom->extent` is defined, then `dom->extent` must have an `Int` datatype and `dtype(dom->extent)` must match `dtype(var)`.
2. `Range(min, extent)`: Always valid.
3. `Buffer(data, dtype, shape, axis_separators, strides, elem_offset, name, data_alignment, offset_factor, buffer_type)`: The following must hold:
    1. «All members of `shape` must have `Int` or `UInt` scalar datatypes.»
    2. «All members of `strides` must have `Int` or `UInt` scalar datatypes and they must all match exactly. (In this specification, we do not permit users to specify strides themselves, so all members must be fresh `Var` nodes.)»
    3. «`elem_offset` must have an `Int` or `UInt` scalar datatype. (In this specification, we do not permit users to specify `elem_offset` themselves, so it must be a fresh `Var` node.)»
    4. `data->type_annotation` must be a `PointerType` and `data->type_annotation->element_type` must be a `PrimType`.
    5. If `buffer_type` is `kAutoBroadcast`, `strides` is empty, and `shape` is nonempty, then treat `strides` as a list of fresh `Var` nodes of the same length as `shape`, where `dtype(strides[i])` matches `dtype(shape[i])` for all `i` from 0 to `len(shape) - 1`.
4. `BufferRegion(buffer, region)`: `region` must be of the same length as `buffer->shape`.
5. `MatchBufferRegion(buffer, source)`: The following must hold:
    1. Let `source_buffer` be `source->buffer`. Let `region` be `source->region`. Let `shape` be `buffer->shape`.
    2. «`buffer->dtype` and `source_buffer->dtype` must match.»
    3. «`source_buffer->data_alignment` must be divisible by buffer->data_alignment.»
    4. «If `buffer->elem_offset` is an `IntImm` with a value of 0, then `source_buffer->elem_offset` must also be an `IntImm` with a value of 0. (In this specification, we do not permit users to specify `elem_offset` themselves, so this condition should not come up.)»
    5. The `buffer->data->storage_scope` must match `source_buffer->data->storage_scope`.
    6. `buffer->buffer_type` and `source_buffer->buffer_type` must be `kDefault`.
    7. Let `offset` be `len(shape) - len(region)`. `offset` must be greater than or equal to 0.
    8. For all `i` from 0 to `offset - 1`, the compiler must be able to statically prove that `region[i]->extent` is numerically equal to 1 (including via arithmetic simplification) or else there is an error.
    9. For all `i` from 0 to `len(shape) - 1`, if `shape[i]` is not a `Var`, the compiler must be able to statically prove that `shape[i]` is numerically equivalent to `region[i + offset]->extent` (including via arithmetic simplification) or else there is an error.
6. `PrimFunc`: If `ret_type` is not defined (note: this is usually the case in practice), treat `ret_type` as `TupleType([])`.

## Semantics

### Variable Scoping

TIR enforces single static assignment (SSA), meaning that all variables must be unique and are bound exactly once. TIR follows lexical scoping, meaning that variables are scoped to the "block" (lexical block, not the `Block` node) in which they are bound: If a variable is in scope, that means it is valid to reference it, and conversely, once it leaves scope, it may no longer be referenced. A list of binding sites:

* `PrimFunc`: Variables that appear in the `params` field are in scope for the entirety of the `PrimFunc` body.
* `Let` and `LetStmt`: The variable in `var` is in scope for the entirety of `body` and leaves scope afterwards.
* `BlockRealize`: Each variable contained in `iter_vars` is in scope when `block` is executed and leaves scope afterwards.
* `For`: `loop_var` is in scope for the entirety of `body` and leaves scope afterwards.
* `Allocate`: The `var` within `buffer_var` is bound for the entirety of `body` and leaves scope afterwards.
* `BufferRealize`: Also acts as an allocation of the buffer, which means it is a binding site for the buffer's `data` field as well as for any fresh variables in the buffer's `stride`, `elem_offset`, and `shape` fields.
* `AttrStmt`: Certain attributes (`thread_extent` and `virtual_thread`) act as binding sites, since they introduce a variable (in the `node` field) that is in scope when the body is executed.
* `Block`: The `Block` acts as a binding site for the buffers (namely their `data` field) mentioned in `alloc_buffers` and `match_buffers`; they leave scope at the end of the `body`. However, the buffers in `alloc_buffers` are not permitted to have unbound `Var`s in their `shape`, `stride`, or `elem_offset` fields, so `alloc_buffers` does not act as a binding site for those variables (any variables in those fields should already be bound).

### State Managed by TIR

TIR programs operate by modifying aspects of the program state. Here is the program state that may be accessed or altered from TIR:
* A TIR program begins by calling a `PrimFunc` that may take as arguments some regions of memory (which are organized in a buffer map). Any buffers in the buffer map are expected to have been allocated by the caller. TIR may access any location within those buffers and may modify the contents of those buffers as well.
* TIR may allocate more memory (i.e., buffers other than those passed as arguments). Any new buffers will also be deallocated by TIR; buffers are generally associated with particular scopes and will be deallocated at the end of the scope.
* Depending on the back-end, TIR statements may also launch new threads and utilize synchronization primitives.
* External calls are capable of invoking arbitrary TVM `PackedFunc`s and can therefore alter any system state. External calls can also invoke device-specific routines that affect the device state. Any usage of external calls to modify state is fully the caller's responsibility to manage; the TIR compiler makes no assumptions about such resources.
* Raising errors or exiting abnormally. This may be done by calls to intrinsics, but also by `AssertStmt`.

In terms of this specification, we consider reads and writes to memory (buffers) and any sort of abnormal exit or I/O side effects to be externally visible, so the semantics for TIR will be described in terms of these actions. For the purposes of the specification, we do not consider memory allocations/deallocations to be directly "observable" by the user, in order to give the compiler greater freedom to rearrange or consolidate memory allocations. Similarly, even though latency and other metrics of performance are very important in practice, the specification does not consider them to be "observable." This provides the compiler with the greatest freedom to make performance-related changes to the code, so long as the other observable behavior remains unchanged. (That is, the specification does not make any promises that specific optimizations are applied in specific situations. The compiler _may_ do those things, but it _must_ preserve the other observable behavior.)

### Buffers

Certain TIR constructs refer to buffers and operations on the memory that underlies them (allocations, accesses, updates, and deallocations). In this version of the specification, we will treat buffers as abstract multidimensional arrays of values of the listed datatype. The specification will thus refer only to the indices of buffers in terms of their shape. A buffer with `dtype` `ty` with shape `(d1, d2, ..., dn)` could thus be conceived of as an array containing `d1` elements, each of which is an array containing `d2` elements, etc., until we finally obtain an array of `dn` elements, each of which is a member of `ty`. 

For example, if the shape is `(2, 2, 3)`, an array representation of it could be `[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]`; index `(0, 1, 1)` would give us element 5 and index `(1, 0, 2)` would give us element 9. 

Note that a buffer can have `()` as a shape; such buffers will be interpreted as storing a single value of the buffer's `dtype`. Each buffer should be assumed to be unique (not aliasing any other), unless it is indicated in the `match_buffers` field of a `Block` (see the semantics for `Block`).

Even though these buffers are represented on real hardware back-ends in terms of memory and operations on buffers are implemented as reads and writes over memory, we will not specify buffers in terms of pointer or indexing arithmetic because details about memory access for hardware back-ends vary greatly (for example, some back-ends use two-dimensional physical indices). Additionally, TIR does not permit direct manipulation of pointers in the language (there is no arithmetic defined for values with the `Handle` datatype). Optimizations like sharing single memory allocations between buffers (e.g., using different offsets or strides) are left to the lower levels of the compiler to implement. Hence, at the top level of TIR, we do not make any guarantees about the representation of buffers in memory, so any TIR code that makes use of such details is not specified by this document.

### Evaluating PrimExprs

`PrimExpr`s in TIR yield values when executed (hence the word "evaluate"). This section describes the value produced by executing each `PrimExpr`.

1. `Var`: If the variable is in scope, return the value bound to that variable. (It is otherwise an error to reference an unbound variable.)
2. `IntImm(value, dtype)`: Evaluates to an integer value equal to `value` with datatype `dtype`.
3. `FloatImm(value, dtype)`: Evaluates to a floating point value equal to `value` with datatype `dtype`.
4. `StringImm(value)`: Evaluates to a `Handle` value. The `Handle` points to the start of a string with the characters of `value`. (The strings are null-terminated C-compatible strings and they are never deallocated.)
5. `Cast(value, dtype)`: Evaluate `value` (calling the result `v`). Cast `v` to datatype `dtype`, returning the cast `value`. If `v` is a vector, then perform the cast element-by-element (assuming no particular ordering). For a scalar or for each element of a vector, casts behave as they would in C (see the ISO C Standard for full formal detail) or in C++'s `static_cast`: unsigned integers are cast by truncating the most significant bits or padding with zeros, casts to signed integers involve sign extension, and casts from floating point to integer truncate. Special cases:
    1. If `value` has a `Handle` datatype, `dtype` must be `Handle`; `Handle` values cannot be cast to other datatypes.
    2. If `dtype` is `Handle` and `value` does not have a `Handle` datatype, this is still valid. The numerical value is cast to a pointer, but the specification makes no guarantees about the result except in the case where `v` is an integer scalar with the value 0: this is treated as a null pointer, which can be used by some builtins.
6. `Select(condition, true_value, false_value)`:
    1. Evaluate `condition` and call the result `c`.
    2. `Select` is not short-circuiting: Evaluate `true_value` and call the result `t`, and evaluate `false_value` and call the result `f`.
    3. If `condition` is a scalar, then if `c` is 1, then the result is `t`. If `c` is 0, the result is `f`.
    4. If `condition` is a vector, then the result will be a vector of the same width. Let us suppose the result is called `r`. For `i` between 0 and `||c|| - 1`, `r.i` is `t.i` if `c.i` is 1 and `f.i` otherwise. No specific order of execution for instantiating the elements of `r` is guaranteed.
7. `BufferLoad(buffer, indices)`:
    1. If `indices` is of length 0, then this means the buffer stores only a single element. In this case, return that single element (do not perform the below steps).
    2. If `indices` is nonempty, evaluate the members of `indices` in order, calling the list of resulting values `indices'`. Let `n` be `len(indices)`.
    3. Cast all values in `indices'` to the integer type expected for the hardware back-end (most commonly, 64-bit unsigned integers as per C's `size_t`, but it may be smaller on some hardware back-ends).
    4. If all members of `indices'` are scalars, then let `v` be the member of `buffer` at index `(indices'[0], indices'[1], ..., indices'[n-1])`.
    5. If `indices'[n-1]` is a vector, let `i_lanes` be its number of lanes. Let `elems` be a list of buffer elements (each of which has a datatype of `buffer->dtype`), of length `i_lanes`. For each `j` from 0 to `i_lanes - 1` (inclusive), `elems[j]` is the element of buffer at the indices `(indices'[0], indices'[1], ..., indices'[n-1].j)`. Let `v` be `concat(elems[0], elems[1], ..., elems[i_lanes-1])`, a single vector with `i_lanes * buffer->dtype->lanes` lanes.
    6. Note that if any set of buffer indices is out of bounds at run time (e.g., if any single member of `indices'` is out of bounds), there is no guarantee on what will result. By default, TIR does not check bounds at run time.
    7. Return `v`.
8. `Ramp(base, stride, lanes)`:
    1. Evaluate `base` and call it `b`. Evaluate `stride` and call it `s`.
    2. The result is a vector with the same bitwidth and typecode as `dtype(b)` with `lanes` for the number of lanes. The `i`th element of the vector is equal to `b + i * s`, following the arithmetical semantics given under the rule for binary operators below (casting `i` to the datatype of `s` if necessary), where `i` ranges from 0 to `lanes - 1` (inclusive).
9. `Broadcast(value, lanes)`: Evaluate `value` and call the result `v` (per the type system, `v` must be a scalar). Return a vector with datatype `DataType(code=dtype(value)->code, bits=dtype(value)->bits, lanes=lanes)`, where all elements of the vector have the value `v`.
10. `Let(var, value, body)`:
    1. Evaluate `value`. Let us call the result `v`.
    2. Create a new scope where `v` is bound to `var`.
    3. Next evaluate `body` in the new scope. Let us call the result `b`.
    4. Pop the scope (i.e., remove `v` from the scope).
    5. Return b.
11. `Call(dtype, op, args)`: This node calls a TIR builtin. All TIR builtins have their own semantics, so no general semantics can be given for `Call(dtype, op, args)`; it is not even guaranteed that the members of args will be evaluated. Instead, see the [section on builtins](#builtin-calls) for a discussion of the semantics of builtins.
12. `Shuffle(vectors, indices)`:
    1. Evaluate the elements of `vectors` in order, calling the list of results `vectors'`.
    2. Evaluate the elements of `indices` in order, calling the list of results `indices'`. Cast the members of indices' to the indexing type expected by the hardware back-end (most commonly a 64-bit unsigned integer, but the width may be smaller on some systems).
    3. Let `v` be the result of concatenating all members of `vectors'` together in order (note: `v` does not have to be materialized in the implementation of this operator, but it allows for specifying the result concisely).
    4. The result is a vector with datatype `DataType(code=dtype(vectors[0])->code, bits=dtype(vectors[0])->bits, lanes=len(indices))`. Let us call the result `r`. For all `i` from 0 to `len(indices) - 1` (inclusive), `r.i` is set to `vectors'[i].indices'[i]`.
13. Binary ops (with arguments `a` and `b`), which are `Add`, `Sub`, `Mul`, `Div`, `Mod`, `FloorDiv`, `FloorMod`, `Min`, and `Max`:
    1. In all cases, evaluate `a` and then `b`, calling the resulting values `v1` and `v2`. Per the type system, these must have the same datatype. For all operators, we will consider a function `f` that describes the semantics of that operator for a single element. If `v1` and `v2` are scalars, then the result will be `f(v1, v2)`, using the below definitions of `f`. If `v1` and `v2` are vectors, then the result will be a vector of the same size, where the `i`th element of the result is `f(v1.i, v2.i)` for all `i` from 0 to `||v1|| - 1`. Note that for computing elements of vectors, no particular order of execution should be assumed. The result of evaluating the expression will have the same datatype as `a` and `b`.
    2. For values with a `Float` typecode, the arithmetic operators below follow the semantics supported by the hardware back-end (generally expected to be IEEE 754, but specialized devices may deviate from it). Analogously, for `BFloat` values, the `bfloat16` specification should be followed. For integers, the operations should be taken to act on the binary representation of the integers (two's complement for signed integers), with the according overflow and underflow behavior as a result (if the bitwidth is `b`, then the max value for unsigned integers is $2^b - 1$ for unsigned integers and for signed integers, the min value is $-(2^{b-1})$ and the max value is $2^{b-1} - 1$).
    3. For `Add`, $f(x, y) = x + y$.
    4. For `Sub`, $f(x, y) = x - y$.
    5. For `Mul`, $f(x, y) = x \cdot y$.
    6. For `Div`, $f(x, y) = x / y$ (where the division is truncating and gives an error for dividing by zero if `v1` and `v2` have `Int` or `UInt` typecodes, otherwise using floating point division if they have `Float` typecodes or Brain float division if they have `BFloat` typecodes).
    7. For `Mod` (only defined for `Int` and `UInt` operands), $f(x, y) = x \text{ mod } y$ (i.e., $x - ((x / y) \cdot y)$ ).
    8. For `FloorDiv`, $f(x, y) = \lfloor x / y \rfloor$.
    9. For `FloorMod`, $f(x, y) = x - (\lfloor x / y \rfloor) \cdot y)$, the remainder of the floor division.
    10. For `Min`, $f(x, y) = \text{min}(x, y)$.
    11. For `Max`, $f(x, y) = \text{max}(x, y)$.
14. Logical ops `And` and `Or`, with arguments `a` and `b`:
    1. If `a` and `b` are scalars, then we implement short-circuiting semantics:
        1. For `And`, evaluate `a` and call the result `v1`. If `v1` is 0, then return 0 (without evaluating `b`). If `v1` is 1, then evaluate `b` and call the result `v2`; return `v2`.
        2. For `Or`, evaluate `a` and call the result `v1`. If `v1` is 1, then return 1 (without evaluating `b`). If `v1` is 0, then evaluate `b` and call the result `v2`; return `v2`.
    2. If `a` and `b` are vectors, then we make _no guarantee as to whether the implementation is short-circuiting on a per-element level_. 
        1. For safety, neither `a` nor `b` should contain side effects (which may happen in calls to builtins); if it is important for there to be side effects, we recommend instead decomposing the vector into scalars.
        2. Suppose that `v1` and `v2` are the result of evaluating `a` and `b`, respectively (though it is not guaranteed that all elements of both will be evaluated). We return a vector of the same size as a where the `i`th element of the vector is `f(v1.i, v2.i)` for each `i` from 0 to `||v1|| - 1`, using the below definitions of `f`:
            1. For `And`, $f(x, y) = x \land y$. 
            2. For `Or`, $f(x, y) = x \lor y$.
15. Logical op `Not` with a unary argument `a`: Evaluate `a` (which must be a `Bool` value, per the type system) and call the result `v`. If `v` is a scalar, return 0 if `v` is 1 and 1 if `v` is 0. If `v` is a vector, return a vector of the same size where the `i`th element is 1 if `v.i` is 0 and 0 if `v.i` is 1, for all `i` from 0 to `||v|| - 1`.
16. Comparison operators (with arguments `a` and `b`), which are `Eq`, `NE`, `LT`, `LE`, `GE`, and `GT`:
    1. In all cases, evaluate `a` and then `b`, calling the resulting values `v1` and `v2`. Per the type system, these must have the same datatype. For all operators, we will consider a function `f` that describes the semantics of that operator for a single element. If `v1` and `v2` are scalars, then the result will be `f(v1, v2)`, using the below definitions of `f`. If `v1` and `v2` are vectors, then the result will be a vector of the same size, where the `i`th element of the result is `f(v1.i, v2.i)` for all `i` from 0 to `||v1|| - 1`. Note that for computing elements of vectors, no particular order of execution should be assumed. The result of evaluating the expression will have the same datatype as `a` and `b`. If `v1` and `v2` have a `Float` typecode, use the semantics supported by the hardware back-end (again, generally expected to be IEEE 754) to determine the results (especially for comparisons with `NaN`, `+inf`, and `-inf`); if they have `Int` or `UInt` typecodes, interpret the comparisons mathematically. Analogously, for `BFloat` values, the `bfloat16` specification should be followed. The datatype of the result is `Bool` in all cases.
    2. For `Eq`, $f(x, y)$ is 1 if $x = y$ (numerically equal) and 0 otherwise.
    3. For `NE`, $f(x, y)$ is 1 if $x \neq y$ (numerically unequal) and 0 otherwise.
    4. For `LE`, $f(x, y)$ is 1 if $x \leq y$ and 0 otherwise.
    5. For `LT`, $f(x, y)$ is 1 if $x < y$ and 0 otherwise.
    6. For `GE`, $f(x, y)$ is 1 if $x \geq y$ and 0 otherwise.
    7. For `GT`, $f(x, y)$ is 1 if $x > y$ and 0 otherwise.

### Builtin Calls

Builtins are external procedures that can be invoked from TIR via the `Call` `PrimExpr`. Note that some TIR documentation and comments refer to builtins as "intrinsics." In this document, we use the term "builtin" to distinguish TIR's builtins from platform-specific intrinsics.

Each builtin in TIR can essentially be treated as a PrimExpr all its own, albeit one that is used too rarely or too situationally to be made a "first-class" part of the AST. TIR builtins generally fit within the following broad categories:
* Platform-specific intrinsics (especially on GPUs)
* Hints to the compiler that have no effect of their own
* Less common mathematical operations
* Interactions with TVM's object system

TIR builtins are also categorized in terms of the effects they may have:
* `kExprAnnotation`: Acts as an annotation for the benefit of the compiler and acts as the identity function for its inputs.
* `kPure`: Acts as a pure function (evaluates its inputs and returns a value, having no other effects).
* `kReadState`: May read memory other than its arguments. For example, this may be global memory or memory that results from dereferencing its arguments (which must also be constructed via builtins).
* `kUpdateState`: May update memory.
* `kOpaque`: Cannot make any assumptions as to whether it reads or writes to any states.
* `kSpecialCallArg`: The intrinsic indicates that its result is a special value that is valid for certain other intrinsic calls. Namely, the result is meant to be used only in a specific context and should not appear outside of that context (e.g., producing a value meant to be used only by certain other builtins).
* `kEmbedInfo`: Acts similarly to `kExprAnnotation`, except the result of its call is removed from the final generated code (i.e., treat the argument as never being evaluated).
* `kControlJump`: Affects control flow.

We will enumerate and describe the builtins in a separate document, as they are added and changed more frequently than other language constructs.

### Semantics of Statements

Unlike `PrimExprs`, statements do not return values. Instead, they operate by modifying the program state. These rules describe how each variety of statement in TIR affects the program state. Statements do, however, depend on values produced by evaluating `PrimExprs`.

1. `PrimFunc(params, body, ret_type, buffer_map, attrs)`: A `PrimFunc` is not technically a statement, but TIR execution always begins by calling a `PrimFunc`.
    1. First, the variables in `params` enter the scope with the called values (passed externally). Note that not all members of `params` need to be passed in as an external argument. Namely, if a variable in `params` appears in the `shape`, `strides`, or `elem_offset` field of any member of `buffer_map`, it will be assigned below, in step ii.
    2. If a member of `params` (let us call it `v`) is a key in `buffer_map`, that means that `v` corresponds to a buffer at run time. The external caller must pass in a pointer to a `DLTensor` (defined in the [`dlpack` library](https://github.com/dmlc/dlpack)); it will correspond to `buffer_map[v]` in the program. Let us refer to the `DLTensor` and `buffer_map[v]` as `t` and `b`, respectively.
        1. The elements of the buffer are read from `t->data` depending on the shape and striding defined. If `t->shape` is empty, then the `DLTensor` stores a single element at location `t->data`; `b` will accordingly also store a single element in the same manner. Otherwise, let `n` be `len(t->shape)` and `S` be the size of a member of `t->dtype`. 
        2. If `t->strides` is null, then `t` has a tightly packed, row-major representation, so the element at `indices` of `b` is at address `data + S*(indices[0]*(t->shape[1]*t->shape[2]*...*t->shape[n-1]) + indices[1]*(t->shape[2]*...*t->shape[n-1]) + ... + indices[n-2]*t->shape[n-1] + indices[n-1])`.
        3. If `t->strides` is not null, then the element `indices` is given by the address `data + S*(indices[0]*t->strides[0] + indices[1]*t->strides[1] + ... + indices[n-1]*t->strides[n-1])`.
        4. Additionally, the following correspondences are checked between `b` and `t`:
            * `b->dtype` and `t->dtype` must match or else there is an error. 
            * Let `elem_offset` be `t->byte_offset` divided by the size of bytes of a member of `b->dtype`. If `b->elem_offset` is a Var, then if it is currently unbound, bind it to `elem_offset`. If `b->elem_offset` is an `IntImm`, its value must match `elem_offset` or else an error is raised. If `b->elem_offset` is a `Var` that is already bound, then its bound value must match `elem_offset` or else an error is raised. 
            * If `t->strides` is null, then `b->strides` must either be empty or it must be of length `t->shape` where `b->strides[i]` is an `IntImm` or bound `Var` with a value of `t->shape[i+1] * t->shape[i+2] * ... * t->shape[n-1]` for all `i` from 0 to `n - 2` and an `IntImm` or bound `Var` with a value of 1 for `b->strides[n - 1]`, where `n` is `len(t->shape)`. (These values for strides are equivalent to having a tightly packed row-major representation.) If neither condition is met, then an error is raised.
            * If `t->strides` is not null, then `len(t->strides)` must match `len(b->strides)` or else an error is raised. For all `i` from 0 to `len(b->strides) - 1`, `b->strides[i]` must be an `IntImm` whose value matches `t->strides[i]` (or else an error is raised), an already bound `Var` whose bound value is `t->strides[i]` (or else an error is raised), or an unbound `Var` (in which case, it is bound with the value `t->strides[i]`). 
            * `len(t->shape)` and `len(b->shape)` must match or else an error is raised. For all `i` from 0 to `len(b->shape) - 1`, `b->shape[i]` must be an `IntImm` whose value matches `t->shape[i]` (or else an error is raised), an already bound `Var` whose bound value is `t->shape[i]` (or else an error is raised), or an unbound `Var` (in which case, it is bound with the value `t->shape[i]`). 
        5. One further condition that `PrimFunc`s expect of their `DLTensor` arguments: No two `DLTensor` arguments are permitted to alias each other. 
        6. Next, `body` is executed. The `PrimFunc` produces outputs by mutating values in buffers passed as the inputs; these changes can be observed by the caller via the `DLTensor` representations passed in step i.
2. `LetStmt(var, value, body)`:
    1. Evaluate `value` (let us call the result `v`). If `var->type_annotation` is a `PointerType`, then implicitly cast `v` to a pointer to `var->type_annotation->element_type` (as far as TIR is concerned, it is simply a `Handle` value).
    2. Push a new scope.
    3. Bind `v` to `var` in the new scope.
    4. Execute `body`.
    5. Pop the scope.
3. `AttrStmt(node, attr_key, value, body)`: For almost all values of `attr_key`, this node has no functional semantics of its own and serves only to provide additional information to the compiler; for those cases, simply evaluate `body`. However, certain values of `attr_key` do affect the semantics and will be described below:
    1. `thread_extent` and `virtual_thread`: These attributes have semantics similar to a parallel `For` loop (though they are realized on hardware in different ways). For these attributes, `node` must be a `Var` node and `value` must be an `IntImm` giving an upper bound.
        1. Evaluate `value` and call the result `v`.
        2. Evaluate `body` `v` times in parallel, binding node to `i` in the `i`th parallel execution. Any interleaving of execution is permitted; additionally, there is no guarantee about how many distinct threads will be created to execute the loop body. If an error occurs in one execution, it is guaranteed that execution will not proceed past the `AttrStmt` but it is not guaranteed that all parallel executions will stop simultaneously or, in the case that multiple executions raise errors, which error will be the one displayed.
        3. `node` is in scope only during the execution of `body` and leaves scope afterwards. `node` cannot be reassigned or modified during the loop body.
    2. `volatile_scope`: In this case, `node` must be a `Var` node. The variable denoted by `node` should be bound somewhere in `body`. This attribute indicates to the compiler that the assignment is volatile in the same sense as the `volatile` keyword in C: The binding and any references to `node` in `body` must not be optimized away by the compiler under any circumstances. The only other semantics is to evaluate `body`.
3. `AssertStmt(condition, message, body)`: Evaluate `condition` (let us call the result `v`). If `v` is 1, then execute `body`. Otherwise, raise an assertion error with `message` as the error message.
4. `BufferStore(buffer, value, indices)`:
    1. Let `buffer_lanes` be `buffer->dtype->lanes`.
    2. Evaluate `value` and call the result `v`. Let `value_lanes` be `dtype(v)->lanes`.
    3. Evaluate `indices` and call the array of results `indices'`. Cast all values in indices' to the integer type expected for the hardware back-end (most commonly, 64-bit unsigned integers as per C's `size_t`, but it may be smaller on some hardware back-ends). Let `n` be `len(indices')`.
    4. Let the number of lanes of `indices'[n-1]` be `i_lanes`.
    5. Depending on `n` and `i_lanes`:
        1. If `n` is 0, then the shape of `buffer` must be `()`. Store `v` as `buffer`'s only element.
        2. If all members of `indices'` are scalars, then store `v` to index `(indices'[0], indices'[1], ..., indices'[n-1])` in `buffer`.
        3. If `i_lanes` is greater than 1, then let `m` be `||v||` and let `W` be `m / buffer->dtype->lanes` (truncating division). For each `j` from 0 to `i_lanes - 1`:
            1. Let `p` be `concat(v.(j*W), v.((j * W) + 1), ..., v.(((j + 1)* W) - 1))`, i.e., take the vector consisting of the `(j*W)`th lane of `v` through the `((j + 1)*W)`th lane of `v` (exclusive), with `W` lanes in total.
            2. Store `p` to element `(indices'[0], indices'[1], ..., indices'[n-2], indices'[n-1].j)` of `buffer`.
        4. Note that if any buffer index is out of bounds at run time, _there is no guarantee on what will result_. By default, TIR does not check bounds at run time.
6. `BufferRealize(buffer, bounds, condition, body)`:
    1. Push a new scope.
    2. Evaluate `buffer->shape`, allocating a new buffer of that shape with the datatype `buffer->dtype` (note: in the compiler implementation, this may be implemented either as an actual allocation or by loading in external data). This acts as an assignment to `buffer->data`.
    3. Evaluate `body`.
    4. Pop the scope and deallocate the newly allocated buffer.
    5. `bounds` and `condition` provide additional information for the compiler for code generation, but do not affect the semantics.
7. `Allocate(buffer_var, dtype, extents, condition, body, annotations)`:
    1. Evaluate `condition` and call the result `c`. If `c` is 0, then finish executing the statement.
    2. Evaluate the members of `extents` in order, calling the list of results `extents'`.
    3. Push a new scope.
    4. Allocate a buffer of shape `extents'` whose entries are of datatype `dtype`. `buffer_var` will be assigned a pointer to this buffer. Note that we do not specify the layout of this buffer in memory. The resulting allocated buffer will not alias any buffer existing in the program. For the purposes of this specification, we assume each allocation to correspond to one and only one buffer; lower levels of compilation may attempt to consolidate memory allocations, but that should not be done at the front end.
    5. Execute `body`.
    6. Deallocate the memory (i.e., delete the buffer) allocated in step iv. Pop the scope.
8. `DeclBuffer(buffer, body)`: Indicates to the compiler that buffer will be in scope for `body`. The only semantics at run time is that `body` is executed.
9. `SeqStmt(seq)`: Execute the statements in `seq` one after the other in the order of the list.
10. `IfThenElse(condition, then_case, else_case)`: Evaluate `condition` (let us call the result `v`). If `v` is 1, execute `then_case`. Otherwise, if `else_case` is present, execute `else_case` (if `else_case` is not present, then do nothing further).
11. `Evaluate(value)`: Simply evaluate `value`, which is a `PrimExpr`. This is effectively a no-op unless `value` contains a call to a builtin, as the result of evaluating `value` cannot be accessed or used by any other statement.
12. `For(loop_var, min, extent, kind, body, thread_binding, annotations)`: The semantics of a `For` statement depend on `kind`:
    1. If `kind` is `kSerial`:
        1. Evaluate `min` and call the result `m`. Cast `m` to the bitwidth of `loop_var`.
        2. Push a new scope.
        3. In the new scope bind `m` to `loop_var`.
        4. Evaluate `extent` and call the result `e`. Cast `e` to the bitwidth of `loop_var`.
        5. If `loop_var` is greater than or equal to `e`, then pop the scope and finish executing the statement.
        6. Evaluate `body`.
        7. Bind `m + 1` to `loop_var`. Return to step c and resume execution from there with this new value of `loop_var`. (Note that `loop_var` cannot be mutated from within the loop body.)
    2. If `kind` is `kParallel`:
         1. Evaluate `min` and call the result `m` and evaluate `extent` and call the result `e`. Let `i1` be `m`, `i2` be `m + 1`, ..., and `in` be `e - 1`.
         2. Evaluate body `e - m` times in parallel, with `loop_var` bound to `ij` in the `j`th parallel execution. Any interleaving of execution is permitted; additionally, there is no guarantee about how many distinct threads will be created to execute the loop body (or whether the executions will actually be in parallel). If an error occurs in one execution, it is guaranteed that execution will not proceed past the `For` statement, but it is not guaranteed that all parallel executions will stop simultaneously or, in the case that multiple executions raise errors, which error will be the one displayed. If any loop iteration writes to a buffer index that is read by any other loop iteration (earlier or later), there is no guarantee on the resulting semantics.
        3. `loop_var` is in scope only during the execution of `body` and leaves scope afterwards. `loop_var` cannot be reassigned or modified during the loop body.
    3. If `kind` is `kVectorized`: The visible semantics are the same as those for `kParallel` in that the loop body will be evaluated `extent` times (`min` must be 0 if `kind` is `kVectorized`); no dependencies between loop iterations are permitted, meaning namely that no loop iteration may write to a buffer index that is read by any other loop iteration (no guarantee is made on the resulting semantics if that is the case). In terms of the implementation, the loop will be implemented by combining loop iterations into single invocations of vectorized operations when this is possible. However, loads and stores to buffers and any other side effects should not be affected by this change—unlike with `kParallel`, the ordering of side effects (including errors) must be preserved.
    4. If `kind` is `kUnrolled`: The semantics are the same as for `kSerial` (this kind simply indicates that the compiler should generate code for the loop by unrolling it rather than including jumps, but it does not change the semantics).
    5. If `kind` is `kThreadBinding`: The semantics are the same as for `kParallel`, but this indicates to the compiler that the loop iterations should be mapped to hardware threads (as in CUDA), with the `thread_binding` field giving further information for the compiler to use for the mapping.
13. `While(condition, body)`:
    1. First, evaluate `condition` (let us call the result `v`).
    2. If `v` is 0, the statement is finished executing.
    3. If `v` is 1, execute `body`. Resume execution from step i.
14. `Block(iter_vars, reads, writes, name_hint, body, init, alloc_buffers, match_buffers, annotations)`:
    1. Push a new scope.
    2. For `i` ranging from 0 to `len(alloc_buffers) - 1`, let us consider the members of `alloc_buffers`:
        1. Evaluate the members of `alloc_buffers[i]->shape`, calling the list of results `shape'`.
        2. Allocate a buffer of datatype `alloc_buffers[i]->dtype` with shape `shape'`, binding the result to `alloc_buffers[i]->data`.
    3. For j ranging from 0 to len(match_buffers) - 1:
        1. Let `buffer` be `match_buffers[j]->buffer`, `source_buffer` be `match_buffers[j]->source->buffer`, and `region` be `match_buffers[j]->source->region`.
        2. In the current scope, we will treat all instance of `buffer` as aliases of `source_buffer`, where every read or write is offset by indices `indices'`, which we define as `[region[0]->min, region[1]->min, ..., region[n-1]->min]`, where `n` is `len(region)`. That is, all `BufferLoad` operations on `buffer` will be be treated as `BufferLoad` operations on `source_buffer` (adding `indices'` to the `BufferLoad`'s indices) and `BufferStore` operations are handled analogously. _This is the **only** form of aliasing permitted in the specification._
            * Note that if `buffer->elem_offset` or `buffer->stride` contain any hitherto unbound variables, they will be bound by pattern matching on `source_buffer`'s run-time representation. Since the specification at this level leaves the element offset and stride as implementation details, it will not be specified how these values are determined (this rule is included to indicate that these variables are bound).
            * `buffer->shape` must have the same length as `region` or else there is an error. For all `i` from 0 to `len(buffer->shape) - 1`, evaluate `region[i]->extent`, which we will call `extent`. `buffer->shape[i]` must be an `IntImm` whose value matches `extent` (or else an error is raised), an already bound `Var` whose bound value is `extent` (or else an error is raised), or an unbound `Var` (in which case, it is bound with the value `extent`).
            * Let `offset` be `len(source_buffer->shape) - len(indices)`. We ignore the first `offset` members of `indices` when doing the offsetting; that is, `buffer` can have fewer dimensions than `source_buffer`. Let `m` be `len(indices)`.
            * For example, given a node `BufferLoad(buffer, indices)`, treat it as `BufferLoad(source_buffer, [indices'[0], indices'[1], ..., indices'[offset-1], Add(indices[0], indices'[offset]), Add(indices[1], indices'[offset+1]), ..., Add(indices[m-1], indices'[n-1])])`. If `indices[m-1]` is a vector rather than a scalar, the last element should instead be `Add(indices[m-1], BroadcastTo(indices'[n-1], ||indices[m-1]||))`.
            * Additionally, no reads or writes may be done on buffer past the indices `[Add(region[0]->min, region[0]->end), Add(region[1]->min, region[1]->end), ..., Add(region[n-1]->min, region[n-1]->end)]` (though, as with out-of-bounds accesses, this is not checked at runtime by default; no semantics are guaranteed for any access outside the bounds listed here).
    4. If any of the vars in `iter_vars` is a reduction `IterVar` (has the type `kCommReduce`), then we consider the block to be a "reduction block." If the block is a reduction block, the block is located in the `body` of a `For` or `While` loop, and `init` is specified, the `init` statement is executed _only during the **first** iteration of the loop_. Otherwise, if specified, the `init` statement is executed each time the block is executed.
    5. Execute `body`. The other fields are included only for the benefit of the compiler in code generation and do not affect the visible semantics.
    6. Pop the scope, deallocating any buffers allocated in step ii.
15. `BlockRealize(iter_values, predicate, block)`:
    1. Open a new scope.
    2. For each `i` from 0 to `len(iter_values) - 1`:
        1. Evaluate `iter_values[i]` and call the result `iter_value`.
        2. Let `iter_var` be `block->iter_vars[i]`.
        3. Bind `iter_value` to `iter_var->var`.
    3. Evaluate `block`. `predicate` provides additional information for the compiler, but does not affect the semantics.
    4. Pop the scope (removing all variables added in step ii).

## TE-Specific Constructs

Some expressions included in the TIR AST implementation are included primarily to support lowering from TE into TIR; they are, however, not part of TIR itself. That is, they will be lowered into other constructs in TIR. These constructs are the following:

* `CommReducer(lhs: [Var], rhs: [Var], result: [PrimExpr], identity_element: [PrimExpr])`
* `Reduce(combiner: CommReducer, source: [PrimExpr], init: [PrimExpr], axis: [IterVar], value_index: int, condition: PrimExpr?)`
* `Prefetch(buffer: Buffer, bounds: [Range])`

## Deprecated Constructs

Some expressions included in the TIR implementation have been deprecated; they should no longer be used and will not be supported. They include the following:
* `Load(dtype: DataType, buffer_var: Var, index: PrimExpr, predicate: PrimExpr)`: Replaced by `BufferLoad`
* `Store(buffer_var: Var, value: PrimExpr, index: PrimExpr, predicate: PrimExpr)`: Replaced by `BufferStore`
* `AllocateConst(buffer_var: Var, data_or_idx: NDArray | IntImm, dtype: DataType, extents: [PrimExpr], body: Stmt, annotations: {str: Object*})`: Never well-supported in the first place. Use an ordinary `Allocate` or an argument to a `PrimFunc` instead.

