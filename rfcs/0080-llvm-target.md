- Feature Name: Encapsulate LLVM target for use with LLVM libraries
- Start Date: May 13, 2022
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub Issue: None

# Summary

1. Enapsulate all information related to a compilation target in LLVM into a
single object `LLVMTarget`. Make creation of this object a prerequisite
for using any LLVM facilities (e.g. optimizations, code generation, etc.).

2. Extend the `llvm` target in TVM to contain LLVM flags, use `LLVMTarget`
to save/restore LLVM's command line options based on the flags contained
in the `llvm` target.

# Motivation

For more details, see [discussion](https://discuss.tvm.apache.org/t/modularizing-llvm-codegen-jit/12764)
on discourse.

The main issue with using statically linked LLVM libraries is that the LLVM
code has, and depends on a global state. A specific (and most problematic)
example of that are command line flags (implemented via `cl::opt` in LLVM).
Many LLVM components use them to tune their behavior, provide debugging or
tracing facilities, or simply as on/off switches. In LLVM sources they are
global variables, and once set they maintain their values.

Since TVM uses LLVM to generate code for multiple different targets, each
specific code generator in TVM may want to use its own set of tuning flags
without affecting code generation for other targets. Similarly, using debug
flag to investigate an issue with a code generation should not affect
unrelated uses of LLVM.

Luckily, LLVM does provide an interface into the command option registry,
which allows clients to query and set the values of these options. TVM
could utilize this to set and restore LLVM's options for the duration of
code generation for each target. This could be done by having a single
"entry point" into LLVM, that each LLVM client would need to use. This
RFC proposes a class that would serve as such entry point.

Since uses of LLVM in TVM are tied to compilation targets, having a common
class describing a compilation target in LLVM's terms would serve two
purposes
1. Would be a unified bridge between the `llvm` target in TVM and target
representation in LLVM, and
2. Would be the "entry point" into LLVM described above.

# Guide-level explanation

The idea of this RFC is to implement a common class `LLVMTarget` for all
LLVM-based targets. Objects of this class would be constructed from TVM's
`Target`, specifically from `Target` objects for `llvm` target.
The objects would contain the LLVM representations of the information
represented by the TVM target, i.e. objects used by LLVM such as
`TargetMachine`. In addition to translating the target data from TVM format
to LLVM format, this object would also query the current state of relevant
LLVM command line options, set then to new values, and restore the original
values on exit (the indent is to use constructor/desctructor for this).

A typical use would follow this pattern:
```C++
{
  // Let's see the LLVM IR and MIR after each transformation, i.e. use
  // -print-after-all in codegen.
  my_target = Target("llvm -mtriple myarch-unknown-elf -llvm-options=print-after-all");
  LLVMTarget llvm_target(my_target);
  // [...]
  // Some uses of llvm_target
  const llvm::Target& t = llvm_target.target_machine->getTarget();
  std::cout << "name: " << t.getName() << "\n";
  std::cout << "description: " << t.getShortDescription() << "\n";
  // [...]
  // Create codegen
  auto cg = new CodeGenMyArch();
  cg->Init(llvm_target);
  // add functions, optimize, save output, etc.
  // [...]
  // Done using LLVM. llvm_target's destructor does the cleanup.
}
```

# Reference-level explanation

## Design considerations

One of the potential further developments could be loading LLVM support
dynamically. Similarly to the saving of LLVM command line options, the call
to dlopen could happen in the constructor of `LLVMTarget`, and the call to
dlclose in its destructor.
This obviously precludes any uses of LLVM outside of the lifetime of the
`LLVMTarget` object, and making it so (or at least coming as close as
possible) was one of the design goals.

There is one case where the `LLVMTarget` object cannot be created before
making use of LLVM: when a LLVM module is deserialized. The `LLVMModule`[1]
class in TVM stores the target string as a metadata in the LLVM IR, and
so the LLVM IR has to be decoded (and the LLVM module created) first,
before a `LLVMTarget` object can be created. To mitigate this issue,
`LLVMTarget` has two "factory" functions, which deserialize an LLVM
module (from file, and from a string), and return a _pair_: the
`LLVMTarget` created from the metadata encoded in the module, and the
LLVM module itself.

Another design consideration was not imposing any limitations on using LLVM,
once the prerequisites were met. In particular, the programmer should be
able to use any LLVM functions or data structures that were available to
them before this proposal.

[1] Unless indicated otherwise, the term "LLVM module" in the text of the RFC
refers to `llvm::Module`. When the name `LLVMModule` is used, it refers to
the TVM type.

## Implementation

One of the more important structures in LLVM, in particular when dealing
with LLVM IR, is `LLVMContext`. A LLVM module needs a context, but it does
not own one. `LLVMContext` should be managed by `LLVMTarget` (in principle,
by anything that outlives the rest of LLVM's objects).

At the minimum, the designed interface would contain:

```C++
class LLVMTarget {
public:
  LLVMTarget(const Target& target);
  ~LLVMTarget();

  std::pair<llvm::Module, LLVMTarget> LoadIR(const std::string& file_name);
  std::pair<llvm::Module, LLVMTarget> ParseIR(const std::string& ir_text);

  std::shared_ptr<llvm::Context> GetOrCreateContext();
};
```

Since the LLVM state is global, there should only be one `LLVMTarget` object
live at any given time if it attempts to modify the state. There can be
arbitrarily many of such objects live simultaneously as long as none of them
modify the state.

# Drawbacks

There is no way to effectively enforce the creation of `LLVMTarget` object
before using LLVM inside TVM, at least not without further steps. This would
have to be a convention that contributors follow, and accidental non-compliance
will not be automatically detected.

# Rationale and alternatives

Having `LLVMTarget` as a prerequisite for using LLVM APIs was intended
to allow its constructor/destructor serve as the "setup"/"cleanup" functions,
similarly to Python's `__enter__` and `__exit__`.  Following Python's `with`
idiom was actually suggested by @tqchen on the discussion forum (thread
linked above).

An alternative way to ensure that LLVM is only used within a certain scope
would be to implement a thin wrapper on top of LLVM, and make all of its APIs
available as members of that wrapper. While adding all available functions
from LLVM as members would be infasible, adding them on an as-needed basis
could actually work. The main reason this approach was not taken is that
one's experience with using LLVM in applications should be directly usable
within TVM, without having to consider additional software layers.

# Prior art

# Unresolved questions

# Future possibilities

With static linking, there is no way to fully save/restore LLVM's global
state. While command line options are the most pressing issue, if there is
a need for further isolation, dynamic loading can be considered. This could
be either loading LLVM libraries built into a shared object, or making the
LLVM as a TVM backend into a shared library.

