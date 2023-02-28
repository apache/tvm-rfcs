Authors: @cloud-mxd, @junrushao,  @tqchen

- Feature Name: Further Unify Packed and Object in TVM Runtime
- Start Date: 2023-01-08
- RFC PR: [apache/tvm-rfcs#0097](https://github.com/apache/tvm-rfcs/pull/97)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

## Summary

This RFC proposes to further unify our PackedFunc and Object in TVM Runtime. Key improvements include: unifying `type_code`, solidifying AnyValue support for both stack and object values, open doors for small-string and NLP-preprocessing, and enable universal container.

## Motivation

FFI is one of the main components of the TVM. We use PackedFunc convention to safely type-erase values and pass things around. In order to support a general set of data structures both for compilation purposes, we also have an Object system, which is made to be aware in the Packed API.

Object supports reference counting, dynamic type casting, and checking as well as structural equality/hashing/serialization in the compiler.
Right now, most of the things of interest are Object, including containers like Map, Array. PackedFunc itself, Module, and various IR objects.
Object requires heap allocation and reference counting, which can be optimized through pooling. They are suitable for most of the deep learning runtime needs,
such as containers, as long as they are infrequent.
In the meantime, we still need to operate with values on the stack. Specifically, when we pass around int, and float values.
It can be wasteful to invoke heap allocations/or even pooling if the operations are meant to be low cost. As a result, the FFI mechanism also serves additional ways to be able to pass **stack values** directly around without object.

This post summarizes lessons from us and other related projects and needs around the overall TVM FFI and Object system. And seek to use these lessons to further solidify the current system. We summarize some of the needs and observations as follows:

### N0: First class stack small string and AnyValue

Data preprocessing is an important part of ML pipeline. Preprocessing in NLP involves strings and containers. Additionally, when translating programs written by users (in python), there may not be sufficient type annotations.

The programs below comes from real production scenario code from matxscript in NLP Preprocessing:

```cpp
// This can be part of data processing code translated
// from user that comes without type annotation
AnyValue unicode_split_any(const AnyValue& word) {
  List ret;
  for (size_t i = 0; i < word.size(); ++i) {
     AnyValue res = word[i];
     ret.push_back(res);
  }
  return ret;
}
// This is a better typed execution code
// Note that word[i] returns a UCS4String container to match python semantics
// Use UCS4String stores Unicode in a fixed-length 4 bytes value to ease random
// access to the elements.
List<UCS4String> unicode_split(const UCS4String& word) {
  List<UCS4String> ret;
  for (size_t i = 0; i < word.size(); ++i) {
     UCS4String res = word[i];
     ret.push_back(res);
  }
  return ret;
}
```
We would like to highlight a few key points by observing the above programs:
- Need a base AnyValue to support both stack values and object.
    - This is to provide a safety net of translation.
- The AnyValue needs to accommodate small-string(on stack) to enable fast string processing. Specifically, note that the particular example creates a `UCS4String res` for every character of the word. If we run heap allocation for each invocation, or even do reference countings, this can become expensive. The same principle also generalizes to the need to accommodate fast processing of other on-stack values.


While it is possible to rewrite the program through stronger typing and get more efficient code. It is important to acknowledge the need to efficient erased runtime support (with minimum overhead), especially given many ML user comes from python.

### N1: Universal Container

In the above example, it is important to note that the container `List` should hold any values. While it is possible to also provide different variant of specialized containers(such as `vector<int>`), to interact with a language like python, it would be nice to have a single universal container across the codebase. We also experienced similar issues in our compilation stack. As an example, while it is possible to use Array to hold IR nodes such as Expr, we cannot use it to hold POD int values, or other POD data types such as DLDataType.

Having an efficient universal container helps to simplify conversions across language as well. For example, a list from python will be able to be turned into a single container without worrying about content type. The execution runtime will also be able to directly leverage the universal container to support all possible cases that a developer might write.

### N2: Further Unify POD Value, Object and AnyValue

TVM currently does have an AnyValue. Specifically `TVMRetValue` is used to hold managed result for C++ PackedFunc return and can serve as AnyValue. Additionally, if the value is an object. `ObjectRef` serves as a nice way that comes with various mechanisms, including structural equality hashing.
We can adopt a process processing called [boxing](https://learn.microsoft.com/en-us/dotnet/csharp/programming-guide/types/boxing-and-unboxing) that enables most of the runtime container to store values as object.
If we create Boxed Object for each stack values, e.g. Integer to represent int. We will be able to effectively represent every value in Object as well.
Both TVMRetValue and Object leverages a code field in the beginning of the data structure to identify the type. TVMRetValue’s code is statically assigned, Object’s code contains a statically assigned segment for runtime objects and dynamically assigned (that are indexed by type_key) for other objects.

There are two interesting regimes of operation that comes with ObjectRef and AnyValue.

- R0: On one hand, if we are operating on the regime of no need for frequent stack value operations. It is desirable to simply use Object. Because object is more compact on register (the size of ptr, which costs 8 bytes on modern 64 bit machines and 4 bytes on 32 bit machines), it can obtain underlying container pointers easily for weak references

    ```cpp
    void ObjectOperation(ObjectRef obj) {
      if (auto* IntImmNode int_ptr = obj.as<IntImmNode>()) {
        LOG(INFO) << int_ptr->value;
      }
    }
    ```

- R1: On the other hand, when we operate on frequent processing that is also not well-typed (as the `unicode_split` example). It is important to also support a AnyValue that comes with stack value support.

As a point of reference, python use object as base for everything. But that indeed creates the overhead for str, int (which we seek to eliminate). Java and C# support both stack values, and their object counter part.
Right now we have both mechanism. It would be **desirable to further unify the Object and AnyValue** to support both R0 and R1. Additionally, it would be nice to have automatic conversions if we decide that two mechanisms are supported. Say a caller pass in a boxed int value, the callee should be able to easily get int out from it(or treat it as an int) without having to do explicit casting. So the same routine can be implemented via either R0 or R1 that is transparent to the caller.

- This is also important for compilers and runtimes, as different compiler and runtime might have their own considerations operating under R0/R1.

## Guide-level explanation and Design Goals

We have the following design goals:

- G0: Automatic switching between object focused scenario and stack-mixed that requires AnyValue.
- G1: Enable efficient string processing, specifically small-string support for NLP use-cases.
- G2: Enable efficient universal container (e.g common container for List/Array that stores everything).
  - Note that it does not prevent us to create specalized code such as `List<String>` as java do, except that
    they still share the same underlying container.
  - Array will share the same container with List to avoid conversion cost.
- G3: Reduce concept duplication(type_code) and provide an unify approach for POD values and object values(including boxing and unboxing)

```cpp
// First class any value
AnyValue unicode_split_any(const AnyValue& word) {
  // universal container
  List ret;
  for (size_t i = 0; i < word.size(); ++i) {
     // efficient small string support
     AnyValue res = word[i];
     ret.push_back(res);
  }
  return ret;
}

// Unify object and POD value handling
// passing an boxed int object to int function and get out int
// automatically without conversion
int MyIntFunc(AnyValue x) {
  int xval = x;
  return x+1;
}

int Caller(Map<String, BoxInt> dict) {
  BoxInt x = dict["x"];
  return MyIntFunc(x);
}
```

Most of the goals are demonstrated in the above example program. We will outline the detailed design in the next section.

## Reference-level Implementation

This section outlines the main design points. We also list design choices and discuss the recommended choices in the rationales and alternative section.

### D0: Key Data Structures

The program below gives an outline of the overall data structure choices.

```cpp

// Object is the same as the current object
// We list it here for reference
struct Object {
  // 4 bytes type code
  // This is a common header with AnyPODBase_
  int32_t type_code;
  // 4 bytes ref counter
  RefCounterType<int32_t> ref_counter;
  // 8 bytes deleter
  typedef void (*FDeleter)(Object* self);
  FDeleter deleter;
  // Rest of the sections.
};

// Common value of Any
struct AnyPODBase_ {
  // type code, this is a common header with Object.
  int32_t type_code;
  // 4 bytes padding can be used to store a number of bytes in small str
  int32_t small_len;
  // 8 bytes field storing variant
  // v_handle can be used to store Object*
  union {
    int64_t v_int64;
    double  v_float64;
    void*   v_handle;
    char    v_bytes[8];
    // UCS4 string and Unicode
    char32_t v_char32[2];
  };
};

// Managed reference of Any value
//Copy will trigger ref counting if
// underlying value is an object.
struct AnyValue : public AnyPodBaseValue_ {
};

// "View" value to any value. Copy will not
// trigger reference counting.
struct AnyView: public AnyPodBaseValue_ {
};

// An any value with extra padding data
// can be used to store larger small str
template<int num_paddings>
struct AnyPad : public AnyValue {
  union {
    char v_pad_bytes[num_paddings * 8];
    // used to support UCS4 string and unicode.
    char32_t v_pad_char32[num_paddings * 2];
  }
};
```

This is a design that outlines the key terms

- T0: Object: the intrusive ptr managed object, used by most containers
    - This is the same as the current object, we list here for clarity.
- T1: AnyValue(aka TVMRetValue): that can stores both pod value and managed reference to ptr
    - By managed reference we mean that copy/destruction of AnyValue will trigger ref counter change if the stored value is an Object
- T2: AnyView(aka TVMArgValue): that stores pod value and un-managed ptr.
- T3: AnyPad: an any value that have larger padded size to accomodate on stack values.
    - When the initial value defaults to null. Both AnyValue and AnyPad, can choose to fill the small_len to be the size of total bytes available. This can help us to be able to pass small string back in C API (without template), by looking at `AnyValue*` ’s small_len field to decide the maximum bytes allowed.

**Discussions**  The default size of AnyValue is 16 bytes. This means that for small string, we can use extra 8 bytes to store the string part(7 bytes if we need a tail `\0`). If we go with UCS4, we can store two extra UCS4 Char without the tail `\0`. The extra space may not be sufficient for some of the small string needs (as a reference matxscript adopts extra padding of 8 bytes to accommodate small string unicode). AnyPad serves as another variation of AnyValue that contains extra stack space. AnyPad is intended to be used interchangeably in any places that AnyValue appears. See also followup sections on conversions function signatures on how that works.

```cpp
// This can be part of data processing code translated
// from user that comes without type annotation
AnyValue unicode_split_any(const AnyValue& word) {
  List ret;
  for (size_t i = 0; i < word.size(); ++i) {
     // we can use AnyPad to store longer small-str
     // in intermediate computation
     AnyPad<1> res = word[i];
     ret.push_back(res);
  }
  return ret;
}
```

Both AnyValue and AnyView also have direct correspondence in the current codebase (TVMRetValue and TVMArgValue). We will use `AnyValue` and `AnyView` for consistency throughout this document.

**Default size of AnyValue** Any variant of AnyPad can be used as default size of AnyValue. For example, we list the following design choices

- **D0a** Default to AnyPad<0> aka 16 bytes. The advantage is smaller size overall in default parameter passing.
- **D0b** Default to AnyPad<1> aka 24 bytes. According to matx’s experience, AnyPad<1> serves well for bytedance’s internal NLP processing needs. However that was also before we had the extra AnyPad proposal. It is now possible to have AnyValue default to 16 bytes, while still create AnyPad during intermediate execution.

**D0str: First-class Small String Handling**

In order to bring first class support for small-string. We adopt the following two kind of type codes.

- kStringObj (managed string object from heap)
- kSmallStr (on-stack small string).

We also need to adopt a String data structure for the in-memory string representation. We can use following code structure (design from the [folly library](https://github.com/facebook/folly))

```cpp
// bytes = std::string = string_core<char>
// str = UCS4String = string_core<char32_t>
// sizeof(string_core) = 24
template <class Char>
class string_core {
  struct MediumLarge {
    Object* data_;  // StringObj
    size_t size_;
  };

  union {
    uint8_t bytes_[sizeof(MediumLarge)];  // For accessing the last byte.
    Char small_[sizeof(MediumLarge) / sizeof(Char)];
    MediumLarge ml_;
  };
  const uint32_t zero_ = 0;            // for c_str
  int32_t category_or_small_len_ = 0;  // small_len: >= 0; large: -2,
};
```

Key elements include:

- There is a zero field to enable `\0` paddings for small-str
- The category_or_small_len field is stored in the end, to accommodate the zero padding
  - When category_or_small_len is bigger than 0, it indicates that it is a small-string with the corresponding length.
  - When category_or_small_len equals -2, it indicates that it belongs to the large string category (that is where the name category comes from).
- For Large string, we will use Object* as the data, which allows us to do reference counting, and direct integration with the object system API.

There will be two objects:

- String: corresponds to std::string, string_core<char>
- UCS4String; string_core<char32_t>

### D1: Unify TypeCode in Object and AnyValue

This is the key idea of this proposal. Right now Object type code and AnyValue type code are separate. We propose to unify them together. The type code will be divided into the following continuous sections (in order):

- **S0:** Special argument passing and POD section
    - kPODIntCode
    - kPODFloatCode
    - kOpaqueHandle
    - ….
    - **kObjectHandle**
    - **kSmallStr**
- **S1:** Special object ptr that can be recognized by minimum TVM runtime.
    - kModule
    - kPackedFunc
    - kNDArray
- **S2:** Boxed object value for Int, Float etc.
- **S3:** Object with static type code.
- **S4:** Object with dynamic type code.

These sections are intentionally made to be continuous. So we can do bound checking to quickly narrow down to a section. Then do switch-case(which can be mapped to a jump table) for in-section specific handling.
By adopting this design, we will have a single, unfied type code throughout the codebase.

- Note that some of the `type_code` (those in S0) **do not** correspond to objects.
- The `type_code` in AnyValue and AnyView can indicate which kind of value it stores, there are two possible design choices here:
    - **D1a:** When `any_value.type_code == kObjectHandle`,  it indicate it is an object in S2-S3, and we can safely lookup the object value, store type_code if it is S0-S1.
    - **D1b:** We can also enforce `any_value.type_code` to be the same as Object.type_code if it stores an object. Note that this will need a type_code lookup  when converting ObjectPtr to any value in S2-S4.
- Some of the `type_code` in S0 may have special meaning for argument passing. For example, TVM supported kTVMObjectRValueRefArg to indicate a move that consumes an object directly without triggering ref counting change (needed for Copy on write and optimize immutable data structure).

One key benefit of unifying the code is that we will be able to store a pointer that is either an `Object*` and `TVMAnyValue*`. This can come handy in universal container design (D3).

```cpp
void Check(void* ptr) {
  int32_t type_code = static_cast<int32_t*>(ptr);
  if (type_code < S0SectionMax) {
    // This is an TVMAnyValue*
  } else {
    // This is an Object*
  }
}
```

**D1section: type code section convention**

One design lesson from matx is that `type_code` in S0 can be represented as negative numbers. That is, we set `S0SectionMax` to be 0.

The main advantage is that it allows backward compatible extensions of both objects(by adding positive numbers) and special sections(by adding negative number).

### D2: Conversion between AnyValue, AnyView and Object

We need to enable universal conversion among the above three kinds of types. In order to do that, we will introduce Boxed object value. Let us discuss the conversion rules between those.

First, conversion between AnyValue and AnyView is reasonably easy.

- AnyValue to AnyView
    - It can simply be a copy if AnyValue == AnyPad<0>
    - If the sequence length is bigger than what AnyView can hold, we need to store it as any_value_ptr (this happens when we pass an AnyPad<n> to AnyView). Specifically, `any view.v_data = &anypad`.
- AnyView to AnyValue
    - Increase ref counter if it stores an object.
    - If we support special value(e.g. C-String passed or Movable object), handle it properly.
- AnyPad<n> to AnyValue
    - When we turn AnyPad<n> to AnyValue(AnyPad<0>), there is a possibility that the stack space in AnyValue cannot hold the small string in AnyPad, in such case, we will turn the string into a boxed string (see also discussion below).
- A pointer to AnyPad<n> can be turned into `AnyValue*`

Let us now discuss how to convert between AnyView/Value and Object. First, the conversion from Any to Object will involve boxing (small-str to String, int to Integer).

- AnyView to Object
    - AnyValue to Object can always be converted to AnyView if needed, or follow some common logics.
    - If the code is in S0, do a switch case and boxing.
    - Special handling code in S1 if there are specific convention.
- AnyView to ObjectPtr<T>
    - This is a case where we can have faster processing if we know T
    - If T is boxed object, run specific conversion logic for T
    - If T contains other objects, check and convert.

The conversion from Object to AnyValue(which can then be converted to AnyView) can have two possibilities:

- **D2a:** Simply keep object as they are when writing to AnyValue ****
    - This simplifies conversion from Object to AnyValue. But when we convert Any into POD values, we will need to check whether if it is Boxed.
- **D2b:** Always unbox to the POD value if the object is a boxed value.
    - This simplifies conversion of AnyValue into POD, since there is no need to check for boxed values.

We encourage D2b when possible, this is because such conversion can be simplified in assignment. It also can help to simplify compiler side logic which only looks at POD type code but cannot handle the Object boxing.

- Object to AnyValue with unboxing
    - Check if code in S2, unbox to the POD value
- ObjectPtr<T> to AnyValue with unboxing
    - The can become a static check which simplifies the logic.

### D3: Universal Container

Now let us discuss the ways to implement universal container that can store stack value and Object.

**D3a: AnyValue as container item**

The first design is simply use AnyValue as the item in the container. This will allow us to store object and AnyValue. Per matx’s experience, to accommodate small str, we might want to allow 24 bytes in the elements. So if we are storing an List that only contains `object`, we only need 24 bytes per object instead of 8bytes. Note that the size of AnyPad can change in the list convention as they are not visible to the users

```cpp
class ListObj {
 private:
  AnyPad<1>* data;
  uint64_t size;
};
```

**D3b: Turn Everything to Object**

We can also turn everything into object*, this would cost 8byte per element, but we will pay boxing cost for small-str and POD values. If we start with unboxed value, then we will pay the Object cost(24bytes) + ptr cost(8bytes). If we start with boxed value, then we pay the ptr cost(8bytes)

```cpp
class ListObj {
 private:
  Object* data;
  uint64_t size;
};
```

**D3c: UnifyItem**

```cpp

// A pool that allocate AnyValue slots so we can store pod values
class AnyValuePool {
 public:
  AnyPad<1>* head;
};
class ListObj {
 private:
  struct UnifyItem {
    union {
       AnyValue* any;
       Object* obj;
       // can be used to access common type_code header of Any and Object
       int32_t* type_code;
     };
   };
  UnifyItem* data;
  uint64_t size;
  AnyValuePool pod_pool;
};
```

In the above design, we stored an unified item ptr that can be either an any pointer or an object pointer. Note that both any and obj have a common type_code header.

- When we insert an Object*, simply treat it as an Object and we can handle managed ref as usual.
- When we insert an AnyValue that is POD or small-str. We copy it to `pod_pool`, and then take address from any_pool and write into data.
- pod_pool are blocked linked list(so the allocated address won’t change).
    - AnyValue contains existing memory that can be used as `next` pointer to maintain free-list.
    - The head of any_pool can contain size.
    - The specific rule of pod pool can also change.

This approach would take extra storage when we store small-string values, extra(8bytes), which is reasonably negligible comparing to sizeof AnyPad<1>(24 bytes). It will have reduced cost when storing objects (mostly same as normal Object arrays).

We can design UnifyItem to have same management rules:

- When it is an Object, trigger ref counting
- When it is POD value, do nothing

Note that returning value of List must be AnyValue, or ObjectRef. We cannot return UnifyItem to outside of the container since internal of pod_pool is local to the container.

The unified List can be used as underlying container for specialized data structures (e.g. List<T>):

- UnifyItem to T:
    - Depends on its type, either use Object to T, or AnyValue to T.
- set T: turn T into UnifyItem:
    - If T is object
        - turn into Object
        - If T is boxed object, turn into Pod
    - If T is pod, turn into Pod

**D3c-variant, further reducing cost of small int and char**

We can further reduce the cost of memory cost in D3c by having a static AnyValue table for common small integers and frequently appearing characters. Specifically, we can allocate a static part of AnyValue pool for Integer in a small range and not allocate from local AnyValue pool.

**Summary of tradeoffs:**

Memory cost:

- D3a: 24 bytes per Object, 24 bytes per POD(small-str)
- D3b: 8 byte per Object, 8 bytes ptr + 24 bytes(boxed object) if original value is not boxed.
- D3c: 8 bytes per Object, 32 bytes per POD(small-str)
- D3c-variant: 8 bytes per Object, 8 bytes per small Integer and char that have static pool.

Accessing efficiency:

D3b may have overhead for small-str and POD. D3a and D3c are small-str friendly.

### D4: PackedFunc Convention

In this section, we revisit the PackedFunc convention under the new context. The update to the C++ PackedFunc API will run as follows:

- TVMArgs now stores ptr of AnyView, TVMArgs[i] will now return AnyView
- Alias: TVMRetValue = AnyValue, TVMArgValue = AnyView

These changes are invisible to the users as long as they use the same source library. We will immediately gain the ability to do universal switching when defining PackedFunc.

- A developer can choose to develop solely on ObjectRef, in this case automatic conversion happens when we turn AnyView and request Object. We expect most of the compiler development to be in this mode.
- A developer can choose to develop runtime functions that contain AnyValue and AnyView to take benefit of stack values. This will have the benefit that intermediate values can store small-str. We anticipate some of the builtin runtime to operate on this mode.
- String will be backed by both small-str and object for efficiency, which we expect to help compiler as well(as there are a lot of small names).
- Both runtime and compiler will be backed by universal container object, which allows us to simplify the automatic conversion in FFI (you take a python tuple and would expect an Array).

There are several design choices in terms of C API convention in light of this new proposal. They will affect the internal data layout of TVMArgs.

- **D4a:** Current C API:

    ```cpp
    int TVMCPackedFunc(PackedFuncHandle handle,
                       int num_args, int* type_codes, TVMValue* values,
                       int* out_tcode, TVMValue* out_value);
    ```

    - This would require packing packing and unpacking the typecode. Note that small string passing wont work because of the lack of the seq_len padding field.
- **D4b:** Combine type code and TVMValue

    ```cpp

    // TVMAnyView and TVMAnyValue follows the same layout as AnyPodBase_
    int TVMCPackedFunc(PackedFuncHandle handle,
                       int num_args, TVMAnyView* args,
                       TVMAnyValue* out);
    ```

    This approach combines code and value into a single entity, this would mean a change of ABI convention in generated code. This approach makes it possible to directly return a small-str without boxing it (however if it is faced in python frontend, likely we still need to box it to str).

    We will introduce adapters to support the old calling convention, which constructs TVMAnyView on the stack then pass things over in the transitioning period. PackedFunc will continue to support the existing TVMArgs and TVMRetValue signature, which adapts
    to the new calling convention.
    ```c++
    class PackedFunc {
     public:
      // old convention
      void CallPacked(TVMArgs args, TVMRetValue* rv);
      // new convention
      void CallPacked(int num_args, AnyView* args, AnyValue* out)
    }
    ```
    Transition of `TVMFuncCall` also happens in two steps.
    - First step the frontend facing APIs such as `TVMFuncCall` will be kept the same, providing an adapter to call into
      the new convention under the hood.
    - Then we will update the frontend implementation, compiler, and runtime to match the new proposed convention.
    The transition into the new convention is mostly mechanical (combining the type_code and value together). In this particular case,
    we favor fastly moving to a new state to reduce overall complexity.

## Prior Arts

This RFC is a further evolution of TVM’s Packed and Object System. We also learn lessons from related projects, such as matxscript, which demonstrates real world use-case motivations for some of the design perspectives.

Matxscript brought a variant of unified system to serve NLP preprocessing needs that are used in real world productions. The key insights includes:
- First class support of small-str.
- Unified type code with negative values as special section.
- Universal container.
- PackedFunc interface with combined TVMAnyView and TVMAnyValue as arguments.


## Rationales and Alternatives

There are several design choices we listed in the above design points, we summarize them here, provide our rationales and recommendations.

### D0a and D0b: Default size of any value

The default size of any value

- **D0a**: 16 bytes
    - AnyView also cost 16bytes. Use AnyPad<n> in locals when necessary
- **D0b**: 24 bytes

Both choices are likely OK and won’t have a big impact. Because we support AnyPad<n> natively in locals (and can also pass AnyPad<n> around) by taking address of it, we would recommend starting with D0a — this also keeps the overall call stack cost consistent with the current design.

### D1a and D1b: Any.type_code for Object

When to store object type code in AnyValue.type_code. Type code segmentation organization

- **D1a**: Stores type_code if the object is in S1
- **D1b**: Stores all type_code (S1-S4)
- **D1c**: Store type of only static segment of types. (S1-S3 but not S4).

The current state starts with D1a. The tradeoff here is again not as critical. One thing to consider is the object code lookup cost when the frontend only recognizes part of the type. Likely starting from D1a is a reasonable pt.

### D1section: Any.type_code section convention

When to store non-object type code in AnyValue.type_code. We have the following code segmentation organization

- **D1section-a**: [0, S0SectionMax)
- **D1section-b**: [INT32_MIN, 0)

Considering that the object code are positive values (uint32), we can restrict their value range to (0, INT32_MAX), which should be sufficient. In this case, D1b can be used to represent Non-Object. The advantage of **D1section-b** is that the we can easily expand the S0 section along with the object without causing future breaking changes.

### D2a and D2b: Autoboxing convention

This have to do with boxed value handling when storing to AnyValue

- **D2a**: Simply keep object as they are when writing to AnyValue.
- **D2b**: Always unbox to the POD value if the object is a boxed value when writing to AnyValue/View.

We would recommend D2b because it simplifies the AnyValue to T logic. It would also simplify implementation of compiler that generate calls which takes AnyView, because if the compiler only expects an int, it does not need to worry about unboxing.

D2b would shift complexity onto ObjectPtr<T> to AnyView conversion. Note that when T is a stronger type(that do not correspond to a boxed type), usually such unboxing checking can be skipped. When T is an object, it would cost us one range check(to see if the type code is in range S2), which is OK.

### D3a, D3b, D3c: Universal Container Choice

This part considers how can we implement the universal container. This part is generally invisible to the developers and mostly serves as dropping replacement(assuming we have auto conversions in D1). The choices are:

- **D3a**: AnyValue as container item
- **D3b**: Turn Everything to Object and use Object* as container Item
- **D3c**: UnifyItem(Union[Object*, AnyValue*]) as container item. Container contains pod_pool.

D3b would be a desirable choice for compilers as it mostly operates on objects, and freq small-str overhead may not be an issue. It cost 8 bytes per Object. The main drawback is that when freq small-str and POD is an issue (as being motivated by matx), then we need a different solution.

Both D3a and D3c should be able to handle small-str and POD issue efficiently.

- D3a works well for runtime handling where small-str and POD efficiency can be an issue if the  It will cost 24 bytes(3x of D3a and D3c) per item.
- D3c’s preserves the overhead of D3a when operating on Object, and cost 32 bytes(1.3x of D3a) when operating on something that is fully small-string and POD.

We would recommend D3c, with a caveat that it is indeed slightly more eng effort(considering the pod_pool). Note that likely we can have a common pod_pool class that generates UnifyItem, and object containers built on top of it.

### D4a and D4b: PackedFunc Convention

- **D4a**: Same API as the current one
- **D4b**: First class support of any value/View

```cpp
int TVMCPackedFunc(PackedFuncHandle handle,
                   int num_args, TVMAnyView* values,
                   TVMAnyValue* out_value);
```

We will go with D4b as it enables first class passing of small-str and full benefit of the AnyValue/object system unification.

## Phasing

This section discusses the implementation strategy of the proposal. The proposal can be implemented in the following phases:

- M0: Architectural change, AnyValue, AnyView, AnyPad,  alias, type_code segmentation.
    - Implement D0a, D1a, D1section-b, D2b and D4b.
    - The code change would mostly be conventional changes.
    - Note that this implies (by intention) change the ABI of packed func. We will update the compiler/runtime to do so.
    - All front-end, compiler, runtime will be updated together to ensure the current testcases continue to pass.
    - We will introduce an adapter to support the TVMArgs during the transition but favors moving to a new state to reduce overall complexity.
- M1: Introduce new string support (with small string)
- M2: Introduce universal container

We believe the overall milestones are positive given the net gain obtained to enable preprocessing and stronger interpolation with ML ecosystem and the community as well. It also opens doors to bring in features in projects like matx so we can enable efficient NLP preprocessing together with ML workload in the same pipeline.

Additionally, Unifying FFI and Object would bring further unification and reduction in our overall code complexity while leveling up the extensibility, so it serves as a strong improvement to the overall quality of the project.

## Drawbacks

The design proposal would involve changes in our runtime system.
This proposal implies (by intention) change the ABI of packed func.
Please see the phasing section on more details about the phasing plan to introduce such change.
The unpacked API in microTVM won't be affected since it follows a different convention.
This is, however, a positive step to further solidify and reduce the overall amount of concepts in the codebase,
further unify packed and object, and simplify and solidify our implementation alongside of AnyValue and Object.

## Unresolved questions

Most of the design points within the scope are being discussed, and there is nothing that we are aware of.

## Future possibilities

The proposal opens doors to enable future NLP preprocessing and better interpolations other applications with TVM.
One interesting future direction point here is that future compilers can choose to try different AnyPad in code generation
and autotune the padding default to the scenario that best fit the application.

## Appendix

### Relevant String methods

Most of the relevant string methods from matxscript, are based on folly library.

```cpp
template <class Char>
class string_core {
  int32_t category() const noexcept;
  // init by char* and size
  string_core(const Char* const data, size_t size, int32_t category);
  // copy/move construct
  string_core(const string_core& rhs);
  ...
  // access data address
  const Char* data() const noexcept;
  Char* data() noexcept;
  // we might remove mutable part to keep things consistent
  // with immutable data structure
  Char* mutableData();

  // get size/cap
  size_t size() const noexcept;
  size_t capacity() const noexcept;

  // change capacity or size
  void shrink(size_t delta);
  void reserve(size_t minCapacity);
  Char* expandNoinit(size_t delta, bool expGrowth = false);
  void push_back(Char c);

  // check unique
  bool isShared() const noexcept;

  void reset() noexcept;

  void copySmall(const string_core&);
  void copyMedium(const string_core&);
  void copyLarge(const string_core&);

  void initSmall(const Char* data, size_t size);
  void initMedium(const Char* data, size_t size);
  void initLarge(const Char* data, size_t size);

  void reserveSmall(size_t minCapacity);
  void reserveMedium(size_t minCapacity);
  void reserveLarge(size_t minCapacity);

  void shrinkSmall(size_t delta);
  void shrinkMedium(size_t delta);
  void shrinkLarge(size_t delta);

  void unshare(size_t minCapacity = 0);
  Char* mutableDataLarge();
};

class String {
public:
  // some methods like std::string or folly FBString
  ...

private:
  string_core<char> store;
};

class UCS4String {
public:
  // some methods like std::string or folly FBString
  ...

private:
  string_core<char32_t> store;
};

```