- Feature Name: microtvm_project_api
- Start Date: 2020-06-09
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

This RFC describes how TVM integrates with build systems for unconventional platforms, such as those
for microcontrollers and for other bare-metal scenarios.

# Motivation
[motivation]: #motivation

Though TVM's primary goal is generating code to implement models from a high-level description,
there are several reasons why a user might want to interact with a platform's build system through
TVM:

1. To perform autotuning. TVM's internal operator implementations are merely templates and rely on
   an automatic search process to arrive at a fast configuration for the template on a given
   platform. This search process requires that TVM iteratively build and time code on the platform.
2. To perform remote model execution. A user may wish to try several different models or schedules
   on a different platform without rewriting the platform-specific code. Users can do this by
   building generated model code against a generic implementation of the microTVM RPC server.
3. To debug model execution remotely. Some aspects of model execution are easy to debug using a
   platform-specific debugger; however, some things, such as analyzing intermediate tensor values,
   are more easily accomplished with TVM-specific tooling. By leveraging the generic microTVM
   RPC server used in (2), TVM can provide such tooling in a platform-agnostic way.

TVM currently supports these use cases through a set of interfaces:
1. `tvm.micro.Compiler`: used to produce binary and library artifacts.
2. `tvm.micro.Flasher`: used to program attached hardware
3. `tvm.micro.Transport`: used to communicate with on-device microTVM RPC server

Thus far, implementations of these interfaces have been made for Zephyr, mBED OS, and for
simulated hardware using a POSIX subprocess. The latter two interfaces have proven to be a
relatively good fit; however, `tvm.micro.Compiler` is difficult to implement because it attempts
to replicate a platform's build system in TVM. TVM does not want to incorporate platform-specific
build logic into its codebase.

This proposal unifies these three interfaces to form a "Project-level" interface, recognizing that
it's typical to interact with unconventional platforms and their build systems at this level. It
simplifies the `Compiler` interaction into a project-level Build, and adds an explicit
`generate_project` method to the interface. These changes remove the need to build components
and drive the link process from TVM.

As a goal, this proposal aims to allow for the same use cases as are currently supported with these
improvements:

1. Integrating more naturally with build systems typical of embedded platforms.
2. Allowing TVM to automatically generate projects  platforms to define automated scripts to build projects

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

TVM can interact with platform SDKs via its **Project API**. Such SDKs are common when working with
non-traditional OS platforms, such as those commonly used in embedded systems (e.g. Arduino, Zephyr,
iOS). Given a platform-specific implementation of this Project API, TVM can:
1. Generate projects that integrate implemented TVM models with generic platform runtime componeents
2. Build those generated projects
3. Program attached hardware
4. Drive remote model execution via the TVM RPC Server interface.

This last capability means that TVM can drive autotuning, remotely perform model inference, and debug
models on non-traditional OS such as Arduino, Zephyr, and mobile platforms such as iOS and Android.

To provide support for a platform, a **template project** is first defined. Template projects are
expected to exist entirely inside a directory and are identified to TVM by the path to the directory
locally. Template projects may live either inside the TVM repository (when they can be included in the
TVM CI) or in other version control repositories. The template project contains at minium an
implementation of the **Project API** inside an executable program known as the
**TVM Project API Server**.

To begin working with a particular platform's Project API implementation, the user supplies TVM with
the path to the top-level directory. TVM launches an instance the Project API Server (found at a
standard location in that directory). TVM communicates with the Project API Server using JSON-RPC
over standard OS pipes.

TVM supplies generated code to the Project API Server using [Model Library Format](0001-model-library-format.md).

Below is a survey of example workflows used with the Project API Server:

## Generating a project

1. The user imports a model into TVM and builds it using `tvm.relay.build`.
2. The user supplies TVM with the path to the template project and a path to a non-existent
   directory where the generated project should live.
3. TVM launches a Project API server in the template project.
4. TVM verifies that the template project is indeed a template by invoking the Project API server
   `server_info_query` method.
5. TVM invokes the Project API server `generate_project` method to generate the new project.

## Building and Flashing

1. The user follows the steps under [Generating a project](#generating-a-project).
2. TVM expects the Project API server to copy itself to the generated project. It launches a
   Project API server in the generated project directory.
3. TVM verifies that the generated project is not a template by invoking the Project API server
   `server_info_query` method. This method also returns `options` that can be used to customize
   the build.
4. TVM invokes the Project API server `build` method to build the project.
5. TVM invokes the Project API server `flash` method to program the attached device. The
   `options` can be used to specify a device serial number.

## Host-driven model inference

1. The user follows the steps under [Generating a project](#generating-a-project).
2. TVM invokes the Project API server `connect_transport` method to connect to the remote on-device
   microTVM RPC server.
3. The microTVM RPC server is attached to a traditional TVM RPC session on the host device.
4. TVM drives inference on-device using the traditional TVM RPC methods. The Project API server
   methods `read_transport` and `write_transport` are used to receive and send data.
5. When the inference session is over, TVM invokes the Project API server method `close_transport`
   to release any underlying I/O resources, and terminates the Project API server.

## AutoTVM

1. The user supplies a kernel to the AutoTVM tuner for search-based optimization.
2. AutoTVM generates a set of task configurations, instantiates each task, and then invokes
   `tvm.build` to produce a `Module` for each instantiated task.
3. AutoTVM produces a Model Library Format archive from the `Module` for each instantiated task.
4. AutoTVM passes the Model Library Format archive to the AutoTVM `runner`. Project API overrides the
   traditional AutoTVM runner by providing a [`module_loader`](#module-loader). The microTVM
   `module_loader` connects to a _supervisor_ TVM RPC server which carries out the microTVM project build
   as part of the TVM RPC `session_constructor`. The following steps occur in the
   `session_constructor`:

    1. The Model Library Format tar is uploaded to the supervisor.
    2. The user supplies a path, on the supervisor, to a template project.
    3. The supervisor `session_constructor` performs the steps under
       [Building and Flashing](#building-and-flashing).
    4. The supervisor `session_constructor` invokes the Project API server `connect_transport` method
       to connect to the remote device. The session constructor registers a traditional TVM RPC
       session on the supervisor, and this session is also used by the AutoTVM runner due to the
       `session_constructor` mechanism.

5. The AutoTVM runner measures runtime as normal.
6. The AutoTVM runner disconnects the session, closing the Project API server on the supervisor.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Project API implementation

The Project API is a Remote Procedure Call (RPC)-type mechanism implemented using
[JSON-RPC](https://www.jsonrpc.org/specification). The client and server are implemented in
`python/tvm/micro/project_api`. Tests are implemented in
`tests/python/unittest/test_micro_project_api.py`.

## Project API interface

The functions that need to be implemented as part of a Project API server are defined on the
`ProjectAPIHandler` class in `python/tvm/micro/project_api/server.py`:

```
class ProjectAPIHandler(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def server_info_query(self) -> ServerInfo:
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_project(self, model_library_format_path : pathlib.Path, standalone_crt_dir : pathlib.Path, project_dir : pathlib.Path, options : dict):
        """Generate a project from the given artifacts, copying ourselves to that project.

        Parameters
        ----------
        model_library_format_path : pathlib.Path
            Path to the Model Library Format tar archive.
        standalone_crt_dir : pathlib.Path
            Path to the root directory of the "standalone_crt" TVM build artifact. This contains the
            TVM C runtime.
        project_dir : pathlib.Path
            Path to a nonexistent directory which should be created and filled with the generated
            project.
        options : dict
            Dict mapping option name to ProjectOption.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def build(self, options : dict):
        """Build the project, enabling the flash() call to made.

        Parameters
        ----------
        options : Dict[str, ProjectOption]
            ProjectOption which may influence the build, keyed by option name.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def flash(self, options : dict):
        """Program the project onto the device.

        Parameters
        ----------
        options : Dict[str, ProjectOption]
            ProjectOption which may influence the programming process, keyed by option name.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def connect_transport(self, options : dict) -> TransportTimeouts:
        """Connect the transport layer, enabling write_transport and read_transport calls.

        Parameters
        ----------
        options : Dict[str, ProjectOption]
            ProjectOption which may influence the programming process, keyed by option name.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def disconnect_transport(self):
        """Disconnect the transport layer.

        If the transport is not connected, this method is a no-op.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def read_transport(self, n : int, timeout_sec : typing.Union[float, type(None)]) -> int:
        """Read data from the transport

        Parameters
        ----------
        n : int
            Maximum number of bytes to read from the transport.
        timeout_sec : Union[float, None]
            Number of seconds to wait for at least one byte to be written before timing out. The
            transport can wait additional time to account for transport latency or bandwidth
            limitations based on the selected configuration and number of bytes being received. If
            timeout_sec is 0, write should attempt to service the request in a non-blocking fashion.
            If timeout_sec is None, write should block until at least 1 byte of data can be
            returned.

        Returns
        -------
        bytes :
            Data read from the channel. Less than `n` bytes may be returned, but 0 bytes should
            never be returned. If returning less than `n` bytes, the full timeout_sec, plus any
            internally-added timeout, should be waited. If a timeout or transport error occurs,
            an exception should be raised rather than simply returning empty bytes.

        Raises
        ------
        TransportClosedError :
            When the transport layer determines that the transport can no longer send or receive
            data due to an underlying I/O problem (i.e. file descriptor closed, cable removed, etc).

        IoTimeoutError :
            When `timeout_sec` elapses without receiving any data.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def write_transport(self, data : bytes, timeout_sec : float) -> int:
        """Connect the transport layer, enabling write_transport and read_transport calls.

        Parameters
        ----------
        data : bytes
            The data to write over the channel.
        timeout_sec : Union[float, None]
            Number of seconds to wait for at least one byte to be written before timing out. The
            transport can wait additional time to account for transport latency or bandwidth
            limitations based on the selected configuration and number of bytes being received. If
            timeout_sec is 0, write should attempt to service the request in a non-blocking fashion.
            If timeout_sec is None, write should block until at least 1 byte of data can be
            returned.

        Returns
        -------
        int :
            The number of bytes written to the underlying channel. This can be less than the length
            of `data`, but cannot be 0 (raise an exception instead).

        Raises
        ------
        TransportClosedError :
            When the transport layer determines that the transport can no longer send or receive
            data due to an underlying I/O problem (i.e. file descriptor closed, cable removed, etc).

        IoTimeoutError :
            When `timeout_sec` elapses without receiving any data.
        """
        raise NotImplementedError()
```

## Project Options

Each Project API server can return `project_options` as part of the `server_info_query` response.
These can be specified by the user to allow them to give platform SDK-specific options to each API
method.

```
ProjectOption = collections.namedtuple('ProjectOption', ('name', 'help'))
```

It's expected that user-facing clients of the Project API could expose these either as command-line
flags or e.g. accepting them via a JSON or YAML file.

## ServerInfo

In response to a `server_info_query`, an API server should return this structure:

```
ServerInfo = collections.namedtuple('ServerInfo', ('platform_name', 'is_template', 'model_library_format_path', 'project_options'))
```

Its members are documented below:
- `platform_name`: A unique slug identifying this API server.
- `is_template`: True when this server lives in a template project. When True, `generate_project` can be called.
- `model_library_format_path`: None when `is_template` is True; otherwise, the path, relative to the API server,
  of the Model Library Format archive used to create this project.
- `project_options`: list of `ProjectOption`, defined above.


## Changes to AutoTVM

There are two changes to AutoTVM needed to interwork with Project API. They are documented in the sections below.

### Build Model Library Format artifacts

At present, the AutoTVM `Builder` creates shared libraries. To interoperate with Project API servers, it needs to
create Model Library Format archives. Currently, only `fcompile` may be given to customize the output format.
`Builder` will accept an additional keyword argument `output_format` which defaults to `so`. When `mlf` is given,
Model Library Format `.tar` will be produced.

### Introducing `module_loader` to the runner

Before TVM measures inference time for a given artifact, it needs to connect to a TVM RPC server and load the
generated code. This process will be abstracted behind `module_loader`. The default implementation is as follows:

```
def default_module_loader(pre_load_function=None):
    """Returns a default function that can be passed as module_loader to run_through_rpc.
    Parameters
    ----------
    pre_load_function : Optional[Function[tvm.rpc.Session, tvm.runtime.Module]]
        Invoked after a session is established and before the default code-loading RPC calls are
        issued. Allows performing pre-upload actions, e.g. resetting the remote runtime environment.
    Returns
    -------
    ModuleLoader :
        A function that can be passed as module_loader to run_through_rpc.
    """

    @contextlib.contextmanager
    def default_module_loader_mgr(remote_kwargs, build_result):
        remote = request_remote(**remote_kwargs)
        if pre_load_function is not None:
            pre_load_function(remote, build_result)

        remote.upload(build_result.filename)
        try:
            yield remote, remote.load_module(os.path.split(build_result.filename)[1])

        finally:
            # clean up remote files
            remote.remove(build_result.filename)
            remote.remove(os.path.splitext(build_result.filename)[0] + ".so")
            remote.remove("")

    return default_module_loader_mgr
```

# Drawbacks
[drawbacks]: #drawbacks

Why should we *not* do this?

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

## Choice of `JSON-RPC`

There were a couple of RPC options considered for this:

1. JSON-RPC.
    Pros:
        - Human-readable encoding
        - Very simple to implement (could be in bash with `jq`)
        - Concise specification
        - Packages in several popular languages
    Cons:
        - Heavyweight encoding
        - No streaming facility
        - Implementations aren't as cohesively authored as gRPC
        - Makes for two RPC implementations checked-in to TVM.

2. gRPC
    Pros:
        - Widely supported, compact encoding
        - Clearly-documented API and good support forums
        - Supports streaming, the most natural way to forward TVM RPC traffic.
    Cons:
        - Requires the use of another Python package
        - Requires the use of an IDL compiler
        - Intended use case (datacenter-scale RPC) is overkill.
        - Makes for two RPC implementations checked-in to TVM.

3. TVM RPC
    Pros:
        - Already exists in TVM
        - Some prior art for session forwarding
        - Binary encoding
    Cons:
        - Binary encoding
        - Impossible to use today without compiling TVM
        - Implementation is designed around TVM's remote inference use cases, and will
          likely change as new demands arise there.

TVM RPC was decided against given the requirement that TVM must be compiled. gRPC was considered,
but ultimately rejected because JSON-RPC can be implemented in a single Python file without adding
the complexities of an IDL compiler.

## Transport functions

When generating projects that perform host-driven inference or autotuning, TVM needs some way to
communicate with the project's microTVM RPC server. Prior to this RFC, TVM included driver code for
various transports (e.g. stdio, PySerial, etc). The Project API places this functionality in the
API server, so that TVM doesn't need to include any transport-specific dependencies (e.g. PySerial)
in its required Python dependencies.

There are a couple of subtle details that were changed when re-implementing this interface in
Project API to reduce the complexity of Project API servers.

### Encapsulating binary data

First, JSON-RPC is an ASCII protocol and as such, binary data can't be transmitted without adding
escape characters. To avoid unreadable and large payloads over the Project API RPC, some encoding
scheme needed to be chosen in order to encapsulate the binary data in the protocol. The desired
properties of the encoding scheme are:

 - Representable in JSON without the need for escapes
 - Compact given the above constraint
 - Easy to encode and decode in languages likely to be used for the API server

Another common place this occurs is in transmitting binary data to and from websites, where an
ASCII alphabet is chosen to represent binary data, and the binary data is translated into the
alphabet. Since there aren't enough ASCII characters to represent all 2^8 == 256 binary values in
one byte, a smaller alphabet is chosen, typically with 64 or 85 characters. This is referred to as
`base64` or `base85`, and the binary data is encoded by modular arithmetic into the smaller
alphabet. Python provides standard support for these via the `base64` module, so the most compact
encoding (`base85`) was chosen from those standards to encode binary data in the Project API.

### Timeouts

The read and write calls have the following interface semantics:

 * `read(n, timeout_sec) -> bytes`. Read `n` bytes, raising `IoTimeoutError` if `timeout_sec`
   elapses before `n` bytes were read. No timeout if `timeout_sec` is None.
 * `write(data, timeout_sec)`. Write `data`, raising `IoTimeoutError` if `timeout_sec` elapses
   before all of `data` was sent. No timeout if `timeout_sec` is None.

This is a departure from the previous interface, which allowed the implementation to read or write
less data than was needed, returning as soon as possible. This was done initially to match the
typical UNIX `read` and `write` semantics, but it turns out this was tricky to implement with a
timeout. The reason for this was as follows:

1. Because these semantics were expected from the interface, TVM always commanded it to read `128`
   bytes, even if less were needed.
2. However, not all libraries obeyed these semantics. In particular, PySerial reads data until the
   timeout occurs, possibly returning less than `n` bytes. The implementation tried to read 1 byte
   until `timeout_sec`, then set `timeout_sec` to the expected time taken to transmit `n - 1` bytes,
   and returned whatever it could within that window.
3. In the case where `timeout_sec` was `None` (e.g. when debugging something else), implementations
   _should_ be quite simple. However, this wasn't the case, because TVM mostly requested more data
   than it actually needed. In this case, implementations using PySerial were forced to return 1
   byte, causing a lot of log spam given the number of round-trips. Also, implementations using
   file descriptors were needlessly complicated, since `select` plus a non-blocking read was needed.

In the new interface, implementers know exactly how much data to read and write, and the deadline
for such operations. This interface is overall easier to implement and aligns it better with
PySerial. Implementations can choose the simplest approach, which is particularly beneficial given
that it will be more brittle to depend on shared Python modules from API Server implementations.

# Prior art
[prior-art]: #prior-art

The current way of integrating with third-party platforms is via the abstractions in the `tvm.micro`
namespace:

1. `tvm.micro.Compiler`: used to produce binary and library artifacts.
2. `tvm.micro.Flasher`: used to program attached hardware
3. `tvm.micro.Transport`: used to communicate with on-device microTVM RPC server

There are several drawbacks to the present scenario:

1. Generally speaking, this interface encourages code from any platform used by TVM to live in the
   TVM tree (the main barrier is the CI and reviews).
2. The `Compiler` interface is not the correct abstraction of a platform's build system.
3. The interface is large and spread across multiple classes, making it difficult to assemble a
   list of tasks needed to support new platforms.
4. The implementation is overly complex, making it diffcult to actually support such platforms.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

1. Is anyone particularly opposed the RPC mechanism used here?
2. Does this seem simple for downstream platforms to implement?
3. Are there missing pieces from this initial implementation we should include?

# Future possibilities
[future-possibilities]: #future-possibilities

In the future, one could consider expanding the API slightly to encompass more platform-specific
tasks. So far, the main use case to consider is library generation. For example, suppose someone
wanted to use `tvmc` to produce only a library (not a full project) compatible with a particular
platform. TVM could include such code in the mainline codebase, or it could rely on a "plugin"
which would implement the Project API. A new method `generate_library` could be added, and
additional metadata could be added to the  `server_info_query` reply to allow the API server to
indicate whether libraries or projects or both could be generated.

Note that the Project API does not currently aim to be a generic plugin interface for TVM. Such a
solution is beyond the scope of this RFC.
