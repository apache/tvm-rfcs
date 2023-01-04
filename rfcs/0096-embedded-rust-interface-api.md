- Feature Name: embedded_rust_interface
- Start Date: 2022-10-04
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/96)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

This RFC outlines a set of additional APIs for the C Runtime to enable direct calling of an [AOT micro entrypoint](https://discuss.tvm.apache.org/t/rfc-utvm-aot-optimisations-for-embedded-targets/9849) using Embedded Rust, aiming to provide parity with the [Embedded C APIs](https://discuss.tvm.apache.org/t/rfc-utvm-embedded-c-runtime-interface/9951).

# Motivation
[motivation]: #motivation

Embedded Rust is an emerging field with a eco-system based around a standard [embedded hardware abstraction layer](https://github.com/rust-embedded/embedded-hal) and Rust's inherent memory safety. In order to run ML models on Embedded Devices written in Rust using TVM, we wanted to build an interface which could be used with pure Rust and without `unsafe` code wherever possible. It is believed this interface moves TVM to the forefront of embedded development by embracing this new technology.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

As much as possible, we aim to provide an idiomatic and `safe` Rust experience for users, this is possible for:
* Running a simple model
* Running a model with workspace pools
* Running a model with constant pools
* Running a model with I/O pools
* Using Rust drivers with the Device API

## Running a model
Users will be able to import a generated crate from within the [Model Library Format archive](https://discuss.tvm.apache.org/t/rfc-tvm-model-library-format/9121) which includes the dependencies that are required for running that model, this can be added to a user application in `Cargo.toml`:

```rust
[dependencies]
tvmgen_ultimate_cat_spotter = { path = "./tvm_archive/crates/ultimate_cat_spotter" }
```

The generated crate provides types for the Model and Workspace implementing any necessary TVM Runtime traits which allow
to write applications generic over the model used:

```rust
mod my_app_logic;
extern crate tvmgen_ultimate_cat_spotter as ultimate_cat_spotter;

fn main() {
  let mut input_data: [i8; 25600] = my_app_logic::create_input();
  let mut output_data: [f32; 12] = my_app_logic::create_output();

  let mut workspace = ultimate_cat_spotter::Workspace::new(
    &mut input_data,
    &mut output_data,
  );

  let mut model = ultimate_cat_spotter::Model::new();
  model.infer(&mut workspace, TVMDevice::CPU);
  
  assert_eq!(output_data, my_app_logic::expected_output());
}
```

## Running a model with workspace pools
This extends the above premise and provides an additional memory pool argument:

```rust
mod my_app_logic;
extern crate tvmgen_ultimate_cat_spotter as ultimate_cat_spotter;

fn main() {
  let mut input_data: [i8; 25600] = my_app_logic::create_input();
  let mut output_data: [f32; 12] = my_app_logic::create_output();
  let mut memory_pool: [u8; 20000] = my_app_logic::create_memory_pool();

  let mut workspace = ultimate_cat_spotter::Workspace::new(
    &mut input_data,
    &mut output_data,
    &mut memory_pool,
  );

  let mut model = ultimate_cat_spotter::Model::new();
  model.infer(&mut workspace, TVMDevice::CPU);
  assert_eq!(output_data, my_app_logic::expected_output());
}
```

## Running a model using constant pools
By utilising macros, users can generate the appropriate constant pools at compilation time:

```rust
mod my_app_logic;
extern crate tvmgen_ultimate_cat_spotter as ultimate_cat_spotter;

fn main() {
  let mut input_data: [i8; 25600] = my_app_logic::create_input();
  let mut output_data: [f32; 12] = my_app_logic::create_output();
  let mut memory_pool: [u8; 20000] = my_app_logic::create_memory_pool(tvmgen_default::constant_pool_data!());

  let mut workspace = ultimate_cat_spotter::Workspace::new(
    &mut input_data,
    &mut output_data,
    &mut memory_pool,
  );

  let mut model = ultimate_cat_spotter::Model::new();
  model.infer(&mut workspace, TVMDevice::CPU);
  assert_eq!(output_data, my_app_logic::expected_output());
}
```

## Running a model with I/O memory pools
This utilises Rust slices to take an input array of bytes and provide Rust access to different sections within the memory pools:

```rust
mod my_app_logic;
extern crate tvmgen_ultimate_cat_spotter as ultimate_cat_spotter;

fn main() {
  let mut memory_pool: [u8; 20000] = my_app_logic::create_memory_pool();

  let mut workspace = ultimate_cat_spotter::Workspace::new(
    &mut memory_pool,
  );

  let mut model_input = workspace.input_data();
  my_app_logic::copy_input_data(model_input);

  let mut model = ultimate_cat_spotter::Model::new();
  model.infer(&mut workspace, TVMDevice::CPU);
  assert_eq!(workspace.output_data(), my_app_logic::expected_output());
}
```

## Rust Device API
In the Rust interface, we provide a similar interface as the C Device API, providing a trait for driver authors to implement:

```rust
trait TVMDevice {
    fn activate();
    fn open();
    fn close();
    fn deactivate();
}
```

This can then be used by a driver author as simply:

```rust
impl TVMDevice for MyDriver { ... }
```

Which can be used as an alternative to `TVMDevice::CPU` in the `run` function:

```rust
mod my_app_logic;
mod woofles_accelerator;
extern crate tvmgen_ultimate_cat_spotter as ultimate_cat_spotter;

fn main() {
  let mut memory_pool: [u8; 20000] = my_app_logic::create_memory_pool();

  let mut workspace = ultimate_cat_spotter::Workspace::new(
    &mut memory_pool,
  );

  let mut model_input = workspace.input_data();
  my_app_logic::copy_input_data(model_input);

  let mut model = ultimate_cat_spotter::Model::new();
  model.infer(&mut workspace, TVMDevice::CPU);
  assert_eq!(workspace.output_data(), my_app_logic::expected_output());
}
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

In order to create a Rust interface, we must first compile the C artefacts and then create `safe` wrappers around the resultant code.

## C Backend Compilation
As the AOT LLVM backend is limited to the C++ runtime, we re-use the C backend with the Rust FFI module to compile the C code at build time - this allows us to re-use existing [Embedded C APIs](https://discuss.tvm.apache.org/t/rfc-utvm-embedded-c-runtime-interface/9951) with Rust wrappers, such that the call to the C function `tvmgen_<model>_run` is replaced by a `safe` Rust variant:

This requires an additional build step in `build.rs` to reference the existing C files using `cc-rs`:

```rust
use cc;
fn main() {
    cc::Build::new()
        .include("../codegen/host/include")
        .include("../runtime/include")
        .file("../codegen/host/src/default_lib0.c")
        .file("../codegen/host/src/default_lib1.c")
        .compile("mlf");
}
```

This is contained within the generated crate so the user is not required to manage this.

## C Wrapper Structs
By wrapping the C types in Rust structs, and exposing a `new` constructor it allows us to take a defined Rust array and translate it into a `void*`. Users of this interface only need to deal with pure Rust:

```rust
/// Input tensors for TVM module "rusty_coffee"
#[repr(C)]
pub struct Workspace {
    input: *mut ::std::os::raw::c_void,
    output: *mut ::std::os::raw::c_void,
}

impl Inputs {
    pub fn new <'a>(
        input: &mut [u8; 100],
        output: &mut [u8; 10],
    ) -> Self {
        Self {
            input: input.as_ptr() as *mut ::std::os::raw::c_void,
            output: output.as_ptr() as *mut ::std::os::raw::c_void,
        }
    }
}
```

## C Wrapper Entrypoint
Similar to the above, using the Rust FFI we can create an entrypoint function which passes the void pointers directly into C FFI from the Rust structs with the appropriate `unsafe` block to prevent user facing code from having to be `unsafe`. The `run` wrapper also checks the return code of TVM and converts it into a [standard Rust `Result` object](https://doc.rust-lang.org/rust-by-example/error/result.html). Distinct from the C interface API, the Rust interface only has a concept of `Workspace` which is more consistent across the invocations. The device also always specified, and implementations can be written to use either CPU or another Device as necessary within application code.

```rust
/// Entrypoint function for TVM module "rusty_coffee"
/// # Arguments
/// * `workspace` - Workspace for model to operate on
pub struct Model {
    pub fn new() {
        return Self;
    }

    pub fn infer(
        self,
        workspace: &mut Workspace,
        device: &mut TVMDevice
    ) -> Result<(), ()> {
        unsafe {
            let ret = tvmgen_rusty_coffee_run(
                {
                    .input = workspace.input
                },
                {
                    .output = workspace.output
                },
            );
            if ret == 0 {
                Ok(())
            } else {
                Err(())
            }
        }
    }
}

#[repr(C)]
struct Inputs {
    input: *mut ::std::os::raw::c_void,
}

#[repr(C)]
struct Outputs {
    output: *mut ::std::os::raw::c_void,
}

extern "C" {
    pub fn tvmgen_rusty_coffee_run(
        inputs: *mut Inputs,
        outputs: *mut Outputs,
    ) -> i32;
}
```

## Rust Device API
Additional interfaces can be added to TVM's rust crate to provide `TVMDevice` with an implementation for CPU named `TVMDevice::CPU`:

```rust
pub struct CPU {}
impl TVMDevice for CPU {
    fn activate() {}
    fn open() {}
    fn close() {}
    fn deactivate() {}
}
```

We also define a `TVMDevice` shim to convert the C pointers in the executor to Rust, such as:
```rust
#[no_mangle]
pub extern TVMDeviceWooflesActivate(device: *TVMDevice) {
    *device.activate();
}
#[no_mangle]
pub extern TVMDeviceWooflesOpen(device: *TVMDevice) {
    *device.open();
}
#[no_mangle]
pub extern TVMDeviceWooflesClose(device: *TVMDevice) {
    *device.close();
}
#[no_mangle]
pub extern TVMDeviceWooflesDeactivate(device: *TVMDevice) {
    *device.deactivate();
}
```

# Drawbacks
[drawbacks]: #drawbacks

This introduces a second embedded API alongside the C API, but fundamentally we want users to be able to create applications in the language that best suits their needs.

Whilst it is an innovation project, this will be supported with best efforts but may be lag in features behind the C interface API.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

- Add LLVM support for Embedded AOT, this would take longer and the user interface would only be marginally improved compared to the time taken to achieve it
- Provide sample applications for Embedded Rust, by using the knowledge TVM has about a model it can generate proper types and guide users when building an application rather than them having to recreate it from [Model Library Format archive](https://discuss.tvm.apache.org/t/rfc-tvm-model-library-format/9121).

# Prior art
[prior-art]: #prior-art

This largely aims to follow the APIs defined for `--interface-api=c` (see: [Embedded C APIs](https://discuss.tvm.apache.org/t/rfc-utvm-embedded-c-runtime-interface/9951)), but wraps each call to allow idiomatic user facing Rust code.

It builds upon the [Model Library Format archive](https://discuss.tvm.apache.org/t/rfc-tvm-model-library-format/9121) to provide a complete package for Rust appication developers with all relevant files.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

- Are we as idiomatic as we can be?
- Is there a better abstraction for devices?

# Future possibilities
[future-possibilities]: #future-possibilities

* Move away from the C codegen and provide an LLVM based solution without the C interface API
* Rust based [Project API](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0008-microtvm-project-api.md) templates.
