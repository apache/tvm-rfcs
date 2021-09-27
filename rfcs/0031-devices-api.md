- Feature Name: C Device API
- Start Date: 02-08-2021
- RFC PR: [apache/tvm-rfcs#31](https://github.com/apache/tvm-rfcs/pull/31)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)


# Summary
[summary]: #summary
This RFC aims to provide an API which can be used by the C runtime to abstract the variety of driver APIs for different platforms. This is specifically catering towards RTOS abstractions for embedded device drivers and aims to implement a subset of the overall Device API with supporting infrastructure to enable future expansion.

# Motivation
[motivation]: #motivation

When using an accelerator, such as the [Arm&reg; Ethos&trade;-U](https://github.com/apache/tvm-rfcs/pull/11), an Embedded Real-Time Operating System (RTOS) will provide a device abstraction to access the device resource. When using these abstractions, TVM needs to understand how to interact with a device for a given platform.

Taking the common example of a UART interface (imagining the accelerator is communicated to via this interface); in Zephyr, this would look similar to:

```c
#include <zephyr.h>
#include <device.h>

struct device *uart_dev = device_get_binding("USART0");

char data[] = "Hello World!\r\n";
uart_tx(uart_dev, data, sizeof(data), 100);
```

Whereas in CMSIS, this would look more similar to:

```c
ARM_DRIVER_USART* uart_dev = &Driver_USART0;
uart_dev->Initialize(NULL);

char data[] = "Hello World!\r\n";
uart_dev->Send(data, sizeof(data)/sizeof(data[0]));
```

In this example, you can see the diversity of RTOS implementations for drivers and why it's required to provide a flexible abstraction to pass devices for micro targets.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

## User App
For each `Target` or external compiler (`kCompiler` `Function` attribute) which is registered as requiring the C Device API, TVM presumes that the RTOS, platform, or user application defines a struct type `tvm_device_<device>_t`, where `<device>` is either the `Target` kind or the external compiler name. The user defined `tvm_device_<device>_t` struct is included by the user who chooses an implementation as appropriate to their application. Notably, to avoid dynamic allocation, the user must provide the `tvm_device_<device>_t` struct and initialise it rather than it being created and setup for them in the API. TVM then expects an implementation of the named functions for each device, examples in the case of the "woofles" accelerator:

```c
typedef void* tvm_device_woofles_t; // Called by User App
int32_t TVMDeviceWooflesInit(tvm_device_woofles_t* tvm_dev, ...); // Called by User App
int32_t TVMDeviceWooflesActivate(tvm_device_woofles_t* tvm_dev); // Called by generated code
int32_t TVMDeviceWooflesOpen(tvm_device_woofles_t* tvm_dev); // Called by generated code
int32_t TVMDeviceWooflesClose(tvm_device_woofles_t* tvm_dev); // Called by generated code
int32_t TVMDeviceWooflesDeactivate(tvm_device_woofles_t* tvm_dev); // Called by generated code
int32_t TVMDeviceWooflesDestroy(tvm_device_woofles_t* tvm_dev); // Called by User App
```

Which is implemented as part of a User App:
```c
#include <tvm/runtime/device.h>
#include <tvm/device/woofles/zephyr.h>

struct device* woofles_zephyr_device = device_get_binding("WOOFLES0");
tvm_device_woofles_t accelerator; // Opaque type for accelerator device
TVMDeviceWooflesInit(&accelerator, woofles_zephyr_device);

struct tvmgen_mynetwork_devices devices {
    .accelerator = accelerator
};

int32_t ret = tvmgen_mynetwork_run(
    ...,
    &devices
);

TVMDeviceDestroy(&accelerator);
```

## Platform Structures
Users can take a implementations from `src/runtime/crt/device` and headers from `include/runtime/crt/device` which maps to their platform device implementation. The simplest definition of `tvm_device_<device>_t` is `void*` as no information is provided to TVM.

```c
typedef tvm_device_woofles_t void*;
```

For RTOS implementations, a structure can be created such as this simple Zephyr wrapper (include/runtime/crt/platform/zephyr.h):

```c
#include <device.h>

typedef struct {
    struct device* dev;
} tvm_device_woofles_t;
```

This enables the OS maximum control over the resources required, allows the user application to consolidate the memory used by the device control structures, and provides the opportunity to craft code in whichever way is most idiomatic for that platform, such as if an additional locking mechanism is required:

```c
#include <device.h>
#include <kernel.h>

typedef struct {
    struct device* dev;
    k_mutex lock;
} tvm_device_woofles_t;
```

## Generic Device API
The majority of the device API calls should be added to the platform-agnostic `<device>.h`:
```c
int32_t TVMDeviceWooflesActivate(tvm_device_woofles_t* tvm_dev); // Called by generated code
int32_t TVMDeviceWooflesOpen(tvm_device_woofles_t* tvm_dev); // Called by generated code
int32_t TVMDeviceWooflesClose(tvm_device_woofles_t* tvm_dev); // Called by generated code
int32_t TVMDeviceWooflesDeactivate(tvm_device_woofles_t* tvm_dev); // Called by generated code
```

These can all be implemented using the user-opaque context `tvm_device_<device>_t*`, enabling the majority of TVM code to be portable between RTOS implementations; importantly this applies to those called within operator functions (see below). The executors are agnostic to the underlying device implementation and simply get passed the relevant device pointer which is then passed to the correct symbol.

## Platform Device API
To allow setting of platform specifics into the opaque struct, these should be defined in the platform header. Alongside the header, an additional file will provide implementations (`src/runtime/crt/device/<device>/<platform>.c`):
```c
int32_t TVMDeviceWooflesInit(tvm_device_t* tvm_dev, struct device* zephyr_dev) {
    tvm_dev->device = zephyr_dev;
}
```
This simple wrapper enables type checking of these functions and defining a clear translation boundary between the underlying OS implementation and TVM.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Entrypoint
The entrypoint API defined in [Embedded C Runtime Interface](https://discuss.tvm.apache.org/t/rfc-utvm-embedded-c-runtime-interface/9951) is augmented with the `tvm_mynetwork_devices` structure which contains implemented `tvm_device_t` `struct`s for each device used by the network:
```
typedef struct {
    struct tvm_device_woofles_t* woofles
} tvmgen_mynetwork_devices;
```

These are re-cast to `void *` when entering the AOT main function to pass it without TIR understanding the struct types.

```c
int32_t tvmgen_mynetwork_run(
    ...,
    struct tvmgen_mynetwork_devices* devices
) {
    tvmgen_mynetwork_run_model(
        ...,
        devices->host,
        devices->accelerator
    );
}
```

## Executor Function
Each operator is provided with a single device object which can be abstracted and passed as the `void* resource_handle`. The main function calls into the device API to setup and teardown resources before and after each operator call.

```c
int32_t tvmgen_mynetwork_run_model(..., device0, device1) {
    TVMDeviceWooflesActivate(device0); // Could reserve or enable certain circuitry ahead of time
    TVMDeviceWooflesActivate(device1); // Could reserve or enable certain circuitry ahead of time

    TVMDeviceWooflesOpen(device0); // Opens resource for use
    operator(device0); // Pass resource_handle to operator
    TVMDeviceWooflesClose(device0); // Close device use

    TVMDeviceWooflesOpen(device1); // Opens resource for use
    operator(device1); // Pass resource_handle to operator
    TVMDeviceWooflesClose(device1); // Close device use

    TVMDeviceWooflesDeactivate(device0); // Turn off the device
    TVMDeviceWooflesDeactivate(device1); // Turn off the device
}
```

It's important to note that memory copies can happen at any point within the executor function, there's no limitation within this RFC as to where those take place.

This is a simple and likely sufficient set of hooks which can be used to manage these device transactions.

## Device API Functions
In the example of Zephyr, devices are already a first class concept so many of the functions will no-op but should synchronisation be required, an example implementation could be:

```c
#include <device.h>

typedef struct {
    struct device* dev;
    k_mutex lock;
} tvm_device_woofles_t;

int32_t TVMDeviceWooflesInit(tvm_device_woofles_t* tvm_dev, struct device* zephyr_dev) {
    k_mutex_init(&tvm_dev->lock);
    tvm_dev->dev = zephyr_dev;
}

int32_t TVMDeviceWooflesActivate(tvm_device_woofles_t* tvm_dev) {}

int32_t TVMDeviceWooflesOpen(tvm_device_woofles_t* tvm_dev) {
    k_mutex_lock(&tvm_dev->lock, K_FOREVER);
}

int32_t TVMDeviceWooflesClose(tvm_device_woofles_t* tvm_dev) {
    k_mutex_unlock(&tvm_dev->lock);
}

int32_t TVMDeviceWooflesDeactivate(tvm_device_woofles_t* tvm_dev) {}

int32_t TVMDeviceWooflesDestroy(tvm_device_woofles_t* tvm_dev) {
    tvm_dev->dev = NULL;
}
```

Whereas for CMSIS, you can use the platform-specific function to encapsulate the API to our imaginary UART accessed accelerator:

```c
typedef struct {
    ARM_DRIVER_USART* dev;
} tvm_device_uart_accel_t;

int32_t TVMDeviceUartAccelInit(tvm_device_uart_accel_t* tvm_dev, ARM_DRIVER_USART* uart_dev) {
    uart_dev->Initialize(NULL);
    tvm_dev->dev = uart_dev;
}

int32_t TVMDeviceUartAccelActivate(tvm_device_uart_accel_t* tvm_dev) {}
int32_t TVMDeviceUartAccelOpen(tvm_device_uart_accel_t* tvm_dev) {}
int32_t TVMDeviceUartAccelClose(tvm_device_uart_accel_t* tvm_dev) {}
int32_t TVMDeviceUartAccelDeactivate(tvm_device_uart_accel_t* tvm_dev) {}

int32_t TVMDeviceUartAccelDestroy(tvm_device_uart_accel_t* tvm_dev) {
    tvm_dev->dev->Uninitialize();
}
```

## Operator Usage
Each operator would be expected to utilise one device structure and be passed that as the `resource_handle` parameter, making the assumption that each operator or variant of an operator is only bound to one device at a time. In the following example it can be seen how a accelerators interface is implemented per platform to take this void pointer and call the platform specific driver code.

```c
// Operator takes opaque resource_handle
int32_t my_operator(..., void* resource_handle) {
    if (TVMDeviceWooflesInvoke(resource_handle, ...ins,outs,params...) != 0) {
        return -1;
    }
}

// Platform implementation
int32_t TVMDeviceWooflesInvoke(tvm_device_woofles_t* tvm_dev) {
    struct device* zephyr_dev = tvm_dev->dev;
    my_accelerator_invoke(
        zephyr_dev,
        ...ins,outs,params...
    );
}
```

These operators would have previously been configured to use the device_api using a `Target` attribute:
```
TVM_REGISTER_TARGET_KIND("ethos-u", kDLCPU)
  .set_attr<Bool>("device_api", true);
```
This attribute in conjuction with the C runtime would enable the function calls listed in this RFC to be emitted. In the case of the C++ runtime, these devices would need to register appropriate C++ Device APIs.

## PrimFunc Resource Handle
This would leverage the exposure of `resource_handle` in [Wiring up the PrimFunc resource_handle](https://github.com/apache/tvm-rfcs/pull/34/files). The executor function would then take the appropriate device argument and pass that to the correct operator based on `kTarget` or `kCompiler` attributes.

## Device Discovery
Initially, devices will be defined by Target name or external compiler name. This means if you mark an operator as needing an external `woofles` compiler it would result in a devices struct such as:

```c
struct tvmgen_my_model_devices {
    tvm_device_woofles_t* woofles
};
```

Which would be passed down to the relevant operators via the executor. This applies similarly to `Target` defined devices.

# Drawbacks
[drawbacks]: #drawbacks

* Current limitations with `Target` and external compilers mean that only one of each name can occur at once using this system, this could equally be in future work.
* The initial assumption is that each operator will be mapped to a single device, this design choice means that fusion across devices will not be possible.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

We could leverage more code generation to generate device structures. It is the authors belief that being able to write small self-contained platform implementations will be easier to understand for both users and developers of TVM.

Another route to take is to treat RTOSes as entirely separate from TVM, requiring them to fully configure resources before passing in the `void*`. This removes TVMs ability to add hooks for resource management such as `open` and `close` which could be used to enable/disable entire pieces of circuitry between operators.

# Prior art
[prior-art]: #prior-art
* Uses the existing `resource_handle` in the TVM code which isn't currently propagated
* Extends the C Interface API to add support for devices
* Resource management using `open`/`close` and `init`/`destroy` alongside opaque handles is a common pattern in C libraries

# Unresolved questions
[unresolved-questions]: #unresolved-questions

Is coupling `Target`s with this Device API to the C Runtime acceptable? From the authors point of view this seems an acceptable trade off given the devices won't function correctly without the flow control.

# Future possibilities
[future-possibilities]: #future-possibilities
This RFC aims to put in place the foundation of the Device API to start abstracting the various RTOS drivers. There are other flows that have been considered as extensions to this.

## Memory Copies
Movement of memory between additional devices which may be unable to communicate directly, this could take the form of simply:

```
// Copy from/to
int32_t TVMDeviceWooflesCopyFrom(tvm_device_t* source, void* destination);
int32_t TVMDeviceUartAccelCopyTo(void* source, tvm_device_t* destination);
```

And be integrated into the flow as follows:

```
TVMDeviceWooflesOpen(device1);
operator(..., device1) {
    // some work where device1 can read from memory directly
    // then the result is copied back
    TVMDeviceWooflesCopyFrom(device1, &buffer);
}
TVMDeviceWooflesClose(device1);

TVMDeviceUartAccelOpen(device2);
operator(..., device2) 
    TVMDeviceUartAccelCopyTo(&buffer, device2);{
    // some which only device2 can see
    TVMDeviceUartAccelCopyFrom(device2, &output);
}
TVMDeviceUartAccelClose(device1);
```

The additional operations here require further thought, but the `Open`/`Close` API wrapper demonstrated supports it as an extension. Moving some of these calls into the executor may also enable asynchronous memories copies from within TVM.

## Device Activation Scheduling
In future, the executor can move the `Activate` and `Deactivate` per-device to optimise circuit activation times.

```c
int32_t tvmgen_mynetwork_run_model(..., device0, device1) {
    // device1 is a bit slow to start so we enable it here
    TVMDeviceWooflesActivate(device1); // Could reserve or enable certain circuitry ahead of time

    TVMDeviceWooflesActivate(device0); // Faster activation of device
    TVMDeviceWooflesOpen(device0); // Opens resource for use
    operator(device0); // Pass resource_handle to operator
    TVMDeviceWooflesClose(device0); // Close device use
    TVMDeviceWooflesDeactivate(device0); // Last use of this device, deactivate it

    TVMDeviceWooflesOpen(device1); // Opens resource for use
    operator(device1); // Pass resource_handle to operator
    TVMDeviceWooflesClose(device1); // Close device use
    TVMDeviceWooflesDeactivate(device1); // Last use of this device, deactivate it
}
```