- Feature Name: PackedFunc as Object
- Start Date: 2022-01-01
- RFC PR: https://github.com/apache/tvm-rfcs/pull/51/
- GitHub Issue: https://github.com/apache/tvm/pull/10032

## 1. Summary

This RFC allows developers to use `PackedFunc` as TVM objects, which completes the last missing step of TVM runtime object system; and stabilizes the `PackedFunc` into a layout-stable TVM object, which makes `PackedFunc` shareable across C++ DLL boundary.

## 2. Motivation

Historically, several fundamental data structures in TVM are not part of the runtime object system, namely `NDArray` (not object), `Module` (not object), `String` (not exist), `Array` (not in runtime), `Map` (not in runtime), `PackedFunc` (not yet an object).

The rationale of the original design is mainly for simplicity, which is desirable for the usecases as a monolithic compiler. As time goes on, the community has come to realize the fact that the object system should be inclusive enough and by design allow more convenient integration with vendor libraries. Therefore, as part of the effort in TVM refactoring and TVM Unity, recent work strives to re-implement these core data structures to be consistent with the runtime object protocol with stable ABI guarantee, and thus could be passed across the DLL boundary.

As the central piece of the TVM ecosystem, this proposal focuses on making `PackedFunc` a TVM object. By doing so, it completes the last missing piece of the object ecosystem, allows TVM containers to carry `PackedFunc`s.

In addition, the original design uses a `std::function` to store callable objects, which is not able to be passed across the DLL boundary. However, this proposal deprecates the original design, and introduces a layout-stable one, which enables `PackedFunc`s to be passed across the DLL boundary to bring convenience to the vendor library integration.


## 3. Guide-level introduction

This is mainly a developer-facing feature, and thus there is no sensible change to the existing functionalities to the end users, who are still supposed to use the same `PackedFunc` API.

Only one major object is introduced, `PackedFuncObj`, a TVM object in the runtime system (detailed in the next section) which is an ABI stable data structure for packed functions that could be shared across language and DLL boundary.

To avoid API misuse from developers, the `PackedFuncObj` cannot be created or manipulated directly, and the specialization of its creation `make_object<PackedFuncObj>` will be deleted for safety. Instead, the developer-facing class `PackedFunc` remains responsible for creating and managing the object, and for properly setting its content.

In the future, it’s possible to incrementally add more information into `PackedFuncObj` to better help debugging and error reporting.

Note: This RFC doesn’t change any of the existing functionality, including C ABI or `PackedFunc`’s C++ API. Any modification to the C ABI is out of scope of this RFC. And this RFC does not create new ABIs, just refactors existing ones.

## 4. Reference-level introduction

As introduced below, the RFC introduces a new class:

```C++
class PackedFuncObj : public runtime::Object {
  using FCallPacked = void(const PackedFuncObj*, TVMArgs, TVMRetValue*);
  FCallPacked* f_call_packed_;
};
```

A templated subclass is introduces to do the type-erasing trick:

```C++
template <typename TCallable>
class PackedFuncSubObj : public PackedFuncObj {
  TCallable callable_;
};
```

The `PackedFuncObj` inherits an intrusive reference counter and an object deleter from the `runtime::Object`. Besides, with the inheritance trick on `PackedFuncSubObj`, the field `callable_` is introduced to store the content of the callable object, which can be a function pointer, a struct/class, an anonymous lambda function or any other object. 

To make the change minimal, `PackedFuncObj` is not designed to be serializable, and doesn’t support TVM’s native reflection. Copying the type-erased object is strictly prohibited for now for simplicity, and instead copying the PackedFunc is implemented as a straightforward increment to the reference counter by 1.

## 5. Drawbacks

Just like every change to the runtime, the proposed change could slightly affect runtime’s binary size. The effect, depending on the compiler, could be positive or negative.

Overall, given that it brings significantly better experience as stated in the previous sections, we believe the benefits outweighs the potential drawback.

## 6. Rationale and alternatives

This refactoring is the last missing piece of effort that brings core data structures of the TVM runtime into the ABI-stable TVM runtime.

Alternatively, one might argue that it’s not important whether `PackedFunc` should be a TVM object or not; however, it significantly brings negative impact when TVM object system is used across the DLL boundary, or putting `PackedFunc` into TVM containers.

## 7. Prior Art

`NDArray` and `Module` are brought into the object system according to [RFC Issue #4286](https://github.com/apache/tvm/issues/4286).

Containers, including `String`, `Array` and `Map`, are discussed in [the forum thread](https://discuss.tvm.apache.org/t/discuss-runtime-array-containers-array-adt-string/4582?u=junrushao1994) and brought into the object system. The String part is introduced by [PR #4628](https://github.com/apache/tvm/pull/4628), Array in [PR #5585](https://github.com/apache/tvm/pull/5585), and Map in [PR #5740](https://github.com/apache/incubator-tvm/pull/5740).

DGL, one of the most popular frameworks for distributed graph neural network training, adopts TVM’s object and FFI system.

## 8. Unresolved questions

This RFC only introduces C++ ABI for invoking a `PackedFunc`, which might have some limitation when linking artifacts compiled by different compilers. In the future, more effort should be invested into the design of a stable C ABI when two `PackedFunc`s come from different TVM runtime.

## 9. Future possibilities

Based on similar metaprogramming tricks, it’s possible to extract the function signatures of `TypedPackedFunc` and to make error reporting more readable.
