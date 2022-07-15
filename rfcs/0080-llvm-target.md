- Feature Name: Create LLVM scope class for use with LLVM libraries
- Start Date: May 13, 2022
- RFC PR: [apache/tvm-rfcs#0083](https://github.com/apache/tvm-rfcs/pull/83)
- GitHub Issue: None

# Summary

1. Create an object `LLVMScope` whose lifetime determines the scope of
availability of LLVM functions (except serializing/deserializing LLVM IR).
2. Enapsulate all information related to a compilation target in LLVM into a
single object `LLVMTarget`.

This will allow extending the `llvm` target in TVM to contain LLVM command
line flags.
The `LLVMTarget` could them be used to save/restore LLVM's command line
options based on the flags contained in the `llvm` target.

# Motivation

For more details, see [discussion](https://discuss.tvm.apache.org/t/modularizing-llvm-codegen-jit/12764)
on discourse.

The main issue with using statically linked LLVM libraries is that the LLVM
code has, and depends on a global state. First of all, LLVM needs to be
initialized by registering all targets before any use. Another (and most
problematic) example of that are command line flags (implemented via `cl::opt`
in LLVM). Many LLVM components use them to tune their behavior, provide
debugging or tracing facilities, or simply as on/off switches. In LLVM
sources they are global variables, and once set they maintain their values.

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

Another consideration is the LLVM context (`llvm::LLVMContext`), which is
a common source of a number of LLVM IR constructs (like types, or constants).
LLVM context is required for creating LLVM IR, and its lifetime must be
enough to contain the lifetimes of any LLVM IR (`llvm::Module` in particular).

Uses of LLVM in TVM generally fall into two categories: (1) loading/writing
LLVM IR (`llvm::Module`), and (2) target-specific functionality like
optimization, or code generation. This RFC proposes two classes:
1. `LLVMScope` class that would initialize LLVM as a whole, and maintain
the LLVM context.
2. `LLVMTarget` class that would be a unified bridge between the `llvm`
target in TVM and target representation in LLVM, and eventually handle
the saving and restoration of LLVM command line flags.

# Guide-level explanation

The idea of this RFC is to implement a common class `LLVMScope` that would
manage LLVM intialization and the LLVM context. It would be able to create
LLVM modules (`llvm::Module`)[1], but not be associated with any specific
target.

The target object `LLVMTarget` would require a scope object, and the lifetime
of the scope object must entirely contain any target object. The target object
would be a common location to access LLVM data structures associated with
compilation target, e.g. target machine (`llvm::TargetMachine`), fast math
flags (`llvm::FastMathFlags`), optimization level (`llvm::CodeGenOpt::Level`),
and so on. Once LLVM flags are added to the `llvm` target, the `LLVMTarget`
object would also save/restore the original values when necessary.

A typical use would follow this pattern:
```C++
{
  // Initialize LLVM.
  LLVMScope llvm_scope;
  // Let's see the LLVM IR and MIR after each transformation, i.e. use
  // -print-after-all in codegen.
  my_target = Target("llvm -mtriple myarch-unknown-elf -llvm-options=print-after-all");
  With<LLVMTarget> llvm_target(llvm_scope, my_target);
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

[1] Unless indicated otherwise, the term "LLVM module" in the text of the RFC
refers to `llvm::Module`.

# Reference-level explanation

## Design considerations

One of the potential further developments could be loading LLVM support
dynamically. Similarly to the saving of LLVM command line options, the call
to dlopen could happen in the constructor of `LLVMScope`, and the call to
dlclose in its destructor.
This obviously precludes any uses of LLVM outside of the lifetime of the
`LLVMScope` object, and making it so (or at least coming as close as
possible) was one of the design goals.

Another consideration is the extent of the impact of command line options
in LLVM. Since they are represented as global variables, they are acccessible
nearly anywhere in the LLVM code (including LLVM IR deserialization).
To completely contain any uses of LLVM flags in the scope of saving/restoring
their default values one would have to save them before making any calls to
LLVM code. This is unfortunately impossible, since LLVM command line flags
will eventually become an attribute of `llvm` target, which in certain cases
can only be created once an LLVM module has been deserialized: LLVM modules
store target string as metadata.

Because of that, saving and restoring of LLVM flags will not apply to
serialization or deserialization of LLVM IR.

Another design consideration was not imposing any limitations on using LLVM,
once the prerequisites were met. In particular, the programmer should be
able to use any LLVM functions or data structures that were available to
them before this proposal.

## Implementation

One of the more important structures in LLVM, in particular when dealing
with LLVM IR, is `LLVMContext`. A LLVM module needs a context, but it does
not own one. `LLVMContext` should be managed by `LLVMScope` (in principle,
by anything that outlives the rest of LLVM's objects).

At the minimum, the designed interface would contain:

```C++
class LLVMScope {
 public:
  LLVMScope();
  ~LLVMScope();

  std::shared_ptr<llvm::LLVMContext> GetContext() const { return ctx_; }

  // Assume the "llvm_ir" parameter contains serialized textual LLVM IR.
  // Parse the IR and return the resulting llvm::Module.
  std::unique_ptr<llvm::Module> ParseIR(const std::string& llvm_ir) const;
  // Load LLVM IR from file given by "file_name", and return the created
  // llvm::Module. The file can contain either the bitcode (i.e. "bc"), or
  // text (i.e. "ll").
  std::unique_ptr<llvm::Module> LoadIR(const std::string& file_name) const;

 private:
  std::shared_ptr<llvm::LLVMContext> ctx_;
};
```

Since the LLVM state is global, there should only be one `LLVMTarget` object
live at any given time if it attempts to modify the state. There can be
arbitrarily many of such objects live simultaneously as long as none of them
modify the state.

# Drawbacks

There is no way to effectively enforce the creation of `LLVMScope` or
`LLVMTarget` objects before using LLVM inside TVM. At the same time adding
these objects to common code (e.g. `CodeGenLLVM`) should prevent accidental
misuse of LLVM.

# Rationale and alternatives

Having `LLVMScope` as a prerequisite for using LLVM APIs was intended
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

