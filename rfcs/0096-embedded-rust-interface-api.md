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

As much as possible, we aim to provide a `safe` Rust experience for users, this is possible for:
* Running a simple model
* Running a model with workspace pools
* Running a model with constant pools
* Running a model with I/O pools

But is not possible for using the [C Device API](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0031-devices-api.md) due to the need for a void pointer when passing the driver object.

## Running a model
Users can reference the generated interface API using the `#[path]` macro in Rust to point to the generated interface in the [Model Library Format archive](https://discuss.tvm.apache.org/t/rfc-tvm-model-library-format/9121); this provides functions to create structures to specify inputs and outputs; structures provide the `new` constructor to take Rust types and map to any internal data structure:

```rust
#[path = "../../codegen/host/src/tvmgen_default.rs"]
mod tvmgen_default;

mod my_app_logic;

fn main() {
  let mut input_data: [i8; 25600] = my_app_logic::create_input();
  let mut output_data: [f32; 12] = my_app_logic::create_output();

  let mut model_input = tvmgen_default::Inputs::new(&mut input_data);
  let mut model_output = tvmgen_default::Outputs::new(&mut output_data);

  tvmgen_default::run(&mut model_input, &mut model_output);
  assert_eq!(output_data, my_app_logic::expected_output());
}
```

## Running a model with workspace pools
This extends the above premise and provides an additional memory pool argument:

```rust
#[path = "../../codegen/host/src/tvmgen_default.rs"]
mod tvmgen_default;

mod my_app_logic;

fn main() {
  let mut input_data: [i8; 25600] = my_app_logic::create_input();
  let mut output_data: [f32; 12] = my_app_logic::create_output();
  let mut memory_pool: [u8; 20000] = my_app_logic::create_memory_pool();

  let mut model_input = tvmgen_default::Inputs::new(&mut input_data);
  let mut model_output = tvmgen_default::Outputs::new(&mut output_data);
  let mut model_workspace = tvmgen_default::WorkspacePools::new(&mut memory_pool);

  tvmgen_default::run(&mut model_input, &mut model_output, &mut model_workspace);
  assert_eq!(output_data, my_app_logic::expected_output());
}
```

## Running a model using constant pools
By utilising macros, users can generate the appropriate constant pools at compilation time:

```rust
#[path = "../../codegen/host/src/tvmgen_default.rs"]
mod tvmgen_default;

mod my_app_logic;

fn main() {
  let mut input_data: [i8; 25600] = my_app_logic::create_input();
  let mut output_data: [f32; 12] = my_app_logic::create_output();
  let mut memory_pool: [u8; 20000] = my_app_logic::create_memory_pool(tvmgen_default::constant_pool_data!());

  let mut model_input = tvmgen_default::Inputs::new(&mut input_data);
  let mut model_output = tvmgen_default::Outputs::new(&mut output_data);
  let mut model_workspace = tvmgen_default::WorkspacePools::new(&mut memory_pool);

  tvmgen_default::run(&mut model_input, &mut model_output, &mut model_workspace);
  assert_eq!(output_data, my_app_logic::expected_output());
}
```

## Running a model with I/O memory pools
This utilises Rust slices to take an input array of bytes and provide Rust access to different sections within the memory pools:

```rust
#[path = "../../codegen/host/src/tvmgen_default.rs"]
mod tvmgen_default;

mod my_app_logic;

fn main() {
  let mut memory_pool: [u8; 20000] = my_app_logic::create_memory_pool();

  let mut model_input = tvmgen_default::WorkspacePools::map_inputs(&mut memory_pool);
  my_app_logic::copy_input_data(model_input);

  tvmgen_default::run(&mut model_workspace);

  let mut model_output = tvmgen_default::WorkspacePools::map_outputs(&mut memory_pool);
  assert_eq!(model_output.output, my_app_logic::expected_output());
}
```

## Utilise C Device API
Here we have to use some `unsafe` Rust as the void pointer for a device is intentionally opaque and thus removes the safety guarantees of the Rust compiler. This `unsafe` code should be minimal:

```rust
#[path = "../../codegen/host/src/tvmgen_default.rs"]
mod tvmgen_default;

mod my_app_logic;

fn main() {
  let mut input_data: [i8; 25600] = my_app_logic::create_input();
  let mut output_data: [f32; 12] = my_app_logic::create_output();
  let mut memory_pool: [u8; 20000] = my_app_logic::create_memory_pool();

  let mut model_input = tvmgen_default::Inputs::new(&mut input_data);
  let mut model_output = tvmgen_default::Outputs::new(&mut output_data);
  
  unsafe {
    let mut device_handle = my_app_logic::get_device();
    let mut model_devices = tvmgen_default::Devices::new(&mut device_handle);
    tvmgen_default::run(&mut model_input, &mut model_output, &mut model_devices);
  }

  assert_eq!(output_data, my_app_logic::expected_output());
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

This builds the existing C code and C interface for reference in Rust using the `ffi` module:

```rust

```

## C Wrapper Structs
By wrapping the C types in Rust structs, and exposing a `new` constructor it allows us to take a defined Rust array and translate it into a `void*`. Users of this interface only need to deal with pure Rust:

```rust
/// Input tensors for TVM module "rusty_coffee"
#[repr(C)]
pub struct Inputs {
    input: *mut ::std::os::raw::c_void,
}

impl Inputs {
    pub fn new <'a>(
        input: &mut [u8; 100],
    ) -> Self {
        Self {
            input: input.as_ptr() as *mut ::std::os::raw::c_void,
        }
    }
}
```

For devices, we can only store the void pointer as it's passed without type information through the C runtime:
```rust
/// Device context pointers for TVM module "rusty_coffee"
#[repr(C)]
pub struct Devices {
    ethos_u: *mut ::std::os::raw::c_void,
}

impl Devices {
    pub fn new <'a>(
        ethos_u: *mut ::std::os::raw::c_void,
    ) -> Self {
        Self {
            ethos_u: ethos_u,
        }
    }
}
```

## C Wrapper Entrypoint
Similar to the above, using the Rust FFI we can create an entrypoint function which passes the void pointers directly into C FFI from the Rust structs with the appropriate `unsafe` block to prevent user facing code from having to be `unsafe`. The `run` wrapper also checks the return code of TVM and converts it into a [standard Rust `Result` object](https://doc.rust-lang.org/rust-by-example/error/result.html).

```rust
/// Entrypoint function for TVM module "rusty_coffee"
/// # Arguments
/// * `inputs` - Input tensors for the module
/// * `outputs` - Output tensors for the module
pub fn run(
    inputs: &mut Inputs,
    outputs: &mut Outputs,
) -> Result<(), ()> {
    unsafe {
        let ret = tvmgen_rusty_coffee_run(
            inputs,
            outputs,
        );
        if ret == 0 {
            Ok(())
        } else {
            Err(())
        }
    }
}

extern "C" {
    pub fn tvmgen_rusty_coffee_run(
        inputs: *mut Inputs,
        outputs: *mut Outputs,
    ) -> i32;
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

This largely aims to follow the APIs defined for `--interface-api=c` (see: [Embedded C APIs](https://discuss.tvm.apache.org/t/rfc-utvm-embedded-c-runtime-interface/9951)), but wraps each call to allow `safe` user facing Rust code.

It builds upon the [Model Library Format archive](https://discuss.tvm.apache.org/t/rfc-tvm-model-library-format/9121) to provide a complete package for Rust appication developers with all relevant files.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

- Are we as `safe` as we can be?
- Is there a better abstraction for devices?

# Future possibilities
[future-possibilities]: #future-possibilities

* Move away from the C codegen and provide an LLVM based solution without the C interface API
* Rust based [Project API](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0008-microtvm-project-api.md) templates.
