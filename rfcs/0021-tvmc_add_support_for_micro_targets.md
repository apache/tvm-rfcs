- Feature Name: tvmc_add_support_for_micro_targets
- Start Date: 2021-10-09
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary

This RFC obsoletes RFC **[0]**.

This RFC depends on RFC **[1]**.

This RFC is about how TVMC (TVM CLI tool) can be extended to support microTVM
targets, considering the variety of platforms supported by microTVM, like Zephyr
and Arduino, and also considering future platforms, taking into account the use
of custom/adhoc platforms provided by users at their convenience.

The interface here proposed relies on the new Project API available on microTVM.

PR [9229](https://github.com/apache/tvm/pull/9229) implements the interface in
question **[2]**.

**[0]** [RFC] TVMC: Add support for µTVM -- https://discuss.tvm.apache.org/t/rfc-tvmc-add-support-for-tvm/9049

**[1]** [RFC][Project API] Extend metadata in ProjectOption -- https://github.com/apache/tvm-rfcs/blob/main/rfcs/0020-project_api_extend_metadata.md

**[2]** [TVMC] Add new micro context -- https://github.com/apache/tvm/pull/9229

# Motivation

Currently if a microTVM user wants to compile, build, flash, and run a model for
a micro target available on a supported microTVM platform -- like Zephyr and
Arduino -- it's necessary to drive the process end-to-end via Python scripts.
This is ok when creating tests and developing or experimenting with new microTVM
code, for example, but for some users that just want to try a new model of
interest on a micro target that way can be slow.

TVMC is a command-line driver for TVM (`tvmc`) that allows users to control the
compilation and execution of models on a plethora of targets, using only the
CLI, however it currently does not support microTVM targets, so it can not drive
a similar workflow (compile, run, etc) on microcontrollers.

This RFC proposes to extend TVMC to support the microTVM targets and platforms
so one can easily use `tvmc` to **compile**, **build** (a device image with the
compiled model of interest), **flash**, and **run** a model on a micro target
using only a command-line interface.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Micro targets have unique requirements which justifies the creation of a new
context in `tvmc` called `micro` to accommodate them.

For instance, a given micro target can exist in more than one supported
platform, like the STM32F746 MCU that can be supported by `Zephyr` and by
`FreeRTOS`. Hence a platform must be selected in addition to the MCU and/or the
board.

Moreover, the build and flash steps, although might not be totally exclusive on
micro targets, are quite specific (like using the toolchain and the flash tool
supported by the platform), so much it justifies them to exist inside the new
micro context, i.e. under `tvmc micro`.

In that sense, the following subcommands are available under `tvmc micro`:

**1. create-project**

**2. build**

**3. flash**

`create-project` purpose is for creating a project directory based on **(a)** an
already compiled model kept in a Model Library Format (MLF) archive and **(b)**
a template associated to a platform, like Zephyr, allowing one to specify
several options specific to the selected platform, like the board name and extra
files to be included in the project directory to will be created based on the
template directory. For convenience, an alias (`create`) will exist for this
subcommand.

`build` can be used for building a firmware image ready to be flashed to the
board specified. For example, if Zephyr platform is selected build a
`zephyr.{elf,hex}` image. Pertinent options for 'build' given the platform
selected as an option under 'build' will become available.

`flash` can be used for effectively flashing, to the selected board, the
built firmware. Again, pertinent options for 'flash' context considering the
platform selected will be made available for the user, like for example options
regarding the serial port number to be used when flashing the image to the
board.

`build` and `flash` subcommads are intended to help with debugging inference
flows built on top of Project API infrastructure. Hence they’re not intended to
replace the user’s workflow (e.g. users can still build and flash independently
of these subcommands).

For all these subcommads (1, 2, and 3) the platform template can be selected
among the default platforms (`zephyr` and `arduino`) or can be a custom one, in
that case the platform template must be selected instead. Upon selecting the
template platform option `-d TEMPLATE_DIR` will become available and one can so
specify the adhoc template dir (which must contain an adhoc Project API server
as per the Project API specification).

A **key-value pair** is used to specify the options specific to the platforms.
If the option is of type 'bool' the values available are 'true' and 'false'.
Details on project option types are described in [RFC-0020](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0020-project_api_extend_metadata.md).

An example for that interface (using Zephyr) follows:

1. `create-project` subcommand:

```bash
$ tvmc micro
usage: tvmc micro [-h] {create-project,build,flash} ...
tvmc micro: error: the following arguments are required: subcommand

$ tvmc micro -h
usage: tvmc micro [-h] {create-project,build,flash} ...

optional arguments:
  -h, --help            show this help message and exit

subcommands:
  {create-project,build,flash}
    create-project      create a project template of a given type or given a template dir.
    build               build a project dir, generally creating an image to be flashed, e.g. zephyr.elf.
    flash               flash the built image on a given micro target.

$ tvmc micro create-project
usage: tvmc micro create-project [-h] [-f] PROJECT_DIR MLF {zephyr,arduino,template} ...
tvmc micro create-project: error: the following arguments are required: PROJECT_DIR, MLF, platform

$ tvmc micro create-project /tmp/x100 ./sine_model.tar zephyr
usage: tvmc micro create-project PROJECT_DIR MLF zephyr [--list-options] -o OPTION=VALUE [OPTION=VALUE ...]
tvmc micro create-project PROJECT_DIR MLF zephyr: error: the following arguments are required: -o

$ tvmc micro create-project /tmp/x100 ./sine_model.tar zephyr --list-options
usage: tvmc micro create-project PROJECT_DIR MLF zephyr [--list-options] -o OPTION=VALUE [OPTION=VALUE ...]

optional arguments:
  --list-options        show all options/values for selected platforms/template.
  -o OPTION=VALUE [OPTION=VALUE ...]
                        extra_files_tar=EXTRA_FILES_TAR
                          if given, during generate_project, uncompress the tarball at this path into the project dir.

                        project_type={aot_demo, host_driven}
                          type of project to generate. (required)

                        west_cmd=WEST_CMD
                          path to the west tool. If given, supersedes both the zephyr_base option and ZEPHYR_BASE environment variable.

                        zephyr_board={mimxrt1050_evk, mps2_an521, nrf5340dk_nrf5340_cpuapp,
                                      nucleo_f746zg, nucleo_l4r5zi, qemu_cortex_r5, qemu_riscv32,
                                      qemu_riscv64, qemu_x86, stm32f746g_disco}
                          name of the Zephyr board to build for. (required)

                        config_main_stack_size=CONFIG_MAIN_STACK_SIZE
                          sets CONFIG_MAIN_STACK_SIZE for Zephyr board.

$ tvmc micro create-project /tmp/x200 ./sine_model.tar zephyr -o zephyr_board=stm32f746g_disco project_type=host_driven
$
```

2. `build` subcommand:

```bash
$ tvmc micro build
usage: tvmc micro build [-h] [-f] PROJECT_DIR {zephyr,arduino,template} ...
tvmc micro build: error: the following arguments are required: PROJECT_DIR, platform

$ tvmc micro build -h
usage: tvmc micro build [-h] [-f] PROJECT_DIR {zephyr,arduino,template} ...

positional arguments:
  PROJECT_DIR           Project dir to build.

optional arguments:
  -h, --help            show this help message and exit
  -f, --force           Force rebuild.

platforms:
  {zephyr,arduino,template}
                        you must selected a platform from the list. You can pass '-h' for a selected platform to list its options.
    zephyr              select Zephyr platform.
    arduino             select Arduino platform.
    template            select an adhoc template.

$ tvmc micro build /tmp/x200 zephyr
usage: tvmc micro build PROJECT_DIR zephyr [--list-options] -o OPTION=VALUE [OPTION=VALUE ...]
tvmc micro build PROJECT_DIR zephyr: error: the following arguments are required: -o

$ tvmc micro build /tmp/x200 zephyr --list-options
usage: tvmc micro build PROJECT_DIR zephyr [--list-options] -o OPTION=VALUE [OPTION=VALUE ...]

optional arguments:
  --list-options        show all options/values for selected platforms/template.
  -o OPTION=VALUE [OPTION=VALUE ...]
                        verbose={true, false}
                          run build with verbose output.

                        zephyr_base=ZEPHYR_BASE
                          path to the zephyr base directory.

                        zephyr_board={mimxrt1050_evk, mps2_an521, nrf5340dk_nrf5340_cpuapp,
                                      nucleo_f746zg, nucleo_l4r5zi, qemu_cortex_r5, qemu_riscv32,
                                      qemu_riscv64, qemu_x86, stm32f746g_disco}
                          name of the Zephyr board to build for. (required)

$ tvmc micro build /tmp/x200 zephyr -o zephyr_board=stm32f746g_disco
$
```

3. `flash` subcommand:

```bash
$ tvmc micro flash
usage: tvmc micro flash [-h] PROJECT_DIR {zephyr,arduino,template} ...
tvmc micro flash: error: the following arguments are required: PROJECT_DIR, platform

$ tvmc micro flash -h
usage: tvmc micro flash [-h] PROJECT_DIR {zephyr,arduino,template} ...

positional arguments:
  PROJECT_DIR           Project dir with a image built.

optional arguments:
  -h, --help            show this help message and exit

platforms:
  {zephyr,arduino,template}
                        you must selected a platform from the list. You can pass '-h' for a selected platform to list its options.
    zephyr              select Zephyr platform.
    arduino             select Arduino platform.
    template            select an adhoc template.

$ tvmc micro flash /tmp/x200 zephyr
usage: tvmc micro flash PROJECT_DIR zephyr [--list-options] -o OPTION=VALUE [OPTION=VALUE ...]
tvmc micro flash PROJECT_DIR zephyr: error: the following arguments are required: -o

$ tvmc micro flash /tmp/x200 zephyr --list-options
usage: tvmc micro flash PROJECT_DIR zephyr [--list-options] -o OPTION=VALUE [OPTION=VALUE ...]

optional arguments:
  --list-options        show all options/values for selected platforms/template.
  -o OPTION=VALUE [OPTION=VALUE ...]
                        zephyr_board={mimxrt1050_evk, mps2_an521, nrf5340dk_nrf5340_cpuapp,
                                      nucleo_f746zg, nucleo_l4r5zi, qemu_cortex_r5, qemu_riscv32,
                                      qemu_riscv64, qemu_x86, stm32f746g_disco}
                          name of the Zephyr board to build for. (required)

$ tvmc micro flash /tmp/x200 zephyr -o zephyr_board=stm32f746g_disco
$
```

## Running a model
[running-a-model]: #running-a-model

As shown above by using `tvmc micro` subcommands `create-project`, `build`, and
`flash` it's possible to get a model flashed and ready to run on a micro target.

Running a model is naturally common to all targets (not exclusive on MCUs), so
commands and options for running a model on microcontrollers are accommodated
under the existing `tvmc run` command. To select a micro target the device
'micro' needs to be specified with `--device micro`. When it happens `PATH`
argument can be used not to select a compiled model module file (like currently
it happens on non-micro targets) but to specify the project directory used for
flashing the model of interest:

```bash

$ tvmc run --device micro -h
usage: tvmc run [-h] [--device {cpu,cuda,cl,metal,vulkan,rocm,micro}] [--fill-mode {zeros,ones,random}] [-i INPUTS] [-o OUTPUTS] [--print-time] [--print-top N] [--profile] [--repeat N] [--number N]
                [--rpc-key RPC_KEY] [--rpc-tracker RPC_TRACKER]
                PATH

positional arguments:
  PATH                  path to the compiled module file or to the project directory (micro devices only).

optional arguments:
  -h, --help            show this help message and exit
  --device {cpu,cuda,cl,metal,vulkan,rocm,micro}
                        target device to run the compiled module. Defaults to 'cpu'

[...]
```

Once both `micro` device and project directory are specified `--list-options`
can be used to list all the options available (optional and required ones) for
running the model on the micro target:

```bash
$ tvmc run --device micro /tmp/x200 --list-options
usage: tvmc run [-h] [--device {cpu,cuda,cl,metal,vulkan,rocm,micro}] [--fill-mode {zeros,ones,random}] [-i INPUTS] [-o OUTPUTS] [--print-time] [--print-top N] [--profile] [--repeat N] [--number N]
                [--rpc-key RPC_KEY] [--rpc-tracker RPC_TRACKER] [--list-options] --options OPTION=VALUE [OPTION=VALUE ...]
                PATH

positional arguments:
  PATH                  path to the compiled module file or to the project directory (micro devices only).

optional arguments:
  -h, --help            show this help message and exit
  --device {cpu,cuda,cl,metal,vulkan,rocm,micro}
                        target device to run the compiled module. Defaults to 'cpu'
  --fill-mode {zeros,ones,random}
                        fill all input tensors with values. In case --inputs/-i is provided, they will take precedence over --fill-mode. Any remaining inputs will be filled using the chosen fill mode. Defaults to 'random'
  -i INPUTS, --inputs INPUTS
                        path to the .npz input file
  -o OUTPUTS, --outputs OUTPUTS
                        path to the .npz output file
  --print-time          record and print the execution time(s)
  --print-top N         print the top n values and indices of the output tensor
  --profile             generate profiling data from the runtime execution. Using --profile requires the Graph Executor Debug enabled on TVM. Profiling may also have an impact on inference time, making it take longer to be generated.
  --repeat N            run the model n times. Defaults to '1'
  --number N            repeat the run n times. Defaults to '1'
  --rpc-key RPC_KEY     the RPC tracker key of the target device
  --rpc-tracker RPC_TRACKER
                        hostname (required) and port (optional, defaults to 9090) of the RPC tracker, e.g. '192.168.0.100:9999'
  --list-options        show all options/values for selected platforms/template.
  --options OPTION=VALUE [OPTION=VALUE ...]
                        gdbserver_port=GDBSERVER_PORT
                          if given, port number to use when running the local gdbserver.

                        nrfjprog_snr=NRFJPROG_SNR
                          when used with nRF targets, serial # of the attached board to use, from nrfjprog.

                        openocd_serial=OPENOCD_SERIAL
                          when used with OpenOCD targets, serial # of the attached board to use.

                        zephyr_base=ZEPHYR_BASE
                          path to the zephyr base directory.

                        zephyr_board={mimxrt1050_evk, mps2_an521, nrf5340dk_nrf5340_cpuapp,
                                      nucleo_f746zg, nucleo_l4r5zi, qemu_cortex_r5, qemu_riscv32,
                                      qemu_riscv64, qemu_x86, stm32f746g_disco}
                          name of the Zephyr board to build for. (required)

```

Already existing options that are quite useful when running or testing a model
become automatically available when running a model on microcontrollers, like,
for instance, `--print-top` and `--fill-mode` options.

For example, one can run the flashed example model shown above this way:

```bash
$ tvmc run --device micro /tmp/x200 --options zephyr_board=stm32f746g_disco --print-top 1 --fill-mode random
[[ 0.        ]
 [-0.48289177]]
$
```

`tvmc run` will take care of opening a transport channel (usually via serial)
with the target device where the model is, then it will write the necessary
input data to the run the model and collect back the output, showing it on the
command-line interface if that is requested.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

The proposed interface relies on Project API `server_info_query` method to
obtain all the options available for a specified platform.

`argparse` is used to build the command-line parser, but a dynamic approach is
used to build a sort of dynamic parser: first one parser is built to parse the
platform type; then, once the platform is parsed from the given command line
a query is made using the Project API `server_info_query` method to obtain all
the options available for the specified platform and the parser is incremented
to parse the options based on the selected platform and subcommand.

By using metadata in ProjectOption associated to every project option returned
by `server_info_query` it's possible to build the second parser to parse the
options by platform and the selected subcommand (like `create-project`, `build`,
and `run`).

The first (main) reason for using that "dynamic parser" mechanism is because
it's possible for the user to supply their own platform (when selecting the
`template` adhoc platform and passing a `--template-dir` directory), so it's not
possible to enumerate all platforms' options beforehand.

The second reason is due to the fact that querying the options for a platform
can take about 0.4 seconds. So with the 2 default platforms currently available
(Zephyr and Arduino) it takes about 1 second to query the options and build a
parser accordingly. Thus with 3 or more default platforms the user would start
to feel some slowness when using `tvmc`.

The current implementation avoids querying the options for all the default
platforms available and so will always spend ~0.4s to query and build the proper
parsers to parse the platform options no matter how many platforms might exist.

## Default project templates

Default project templates must be detected and made available as `platform`
options automatically.

The mechanism for automatically detecting the default template directories must
consider basically two different environments: when TVM is built from sources
and when TVM is installed via a Python package (like TVM tlcpack).

The mechanism here suggested is one similar to the one used by
`tvm.micro.get_standalone_crt_dir()`, that relies on
` _ffi.libinfo.find_lib_path()` and so can correctly find the libraries taking
into account if TVM being used is from sources or from a installed package.

Hence for the specific case of the template directories a new function named
`tvm.micro.get_template_dirs()` which returns all available default platform
template directories will be created. The templates themselves will be added to
the wheel package accordingly, just like the `standalone_crt/` is added.

# Drawbacks
[drawbacks]: #drawbacks

`argparse` unfortunately only supports two mutually exclusive groups through
`add_mutually_exclusive_group()`. This complicates a smooth integration of new
micro target specific options into the already existing `tvmc run` command.

For example, `--rpc-key` and `--rpc-tracker` do not apply to micro targets, but
can't be automatically excluded by `argparse` if micro target specific option
`--list-options` is used. This maybe can be solved by enforcing the mutual
exclusive groups in the code instead of in the parser as it's done in the
current implementation and mark these options with something like `(not
available for micro targets)`, just as required options are marked with
`(required)`.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

1. Currently required options that are passed when creating a project, i.e.
passed to `create-project`, need to be repeated for `build`, `flash`, and `run`.
This is ok if one, for whatever reason, wants to change it from one subcommand
to another, but might be a burden to the user that will need to type the
key-value again for each subsequent subcommand. Maybe the required values passed
when `create-project` is used could be recorded in the new project directory
(in a JSON file -- `.tvmc.json`) and a flag called `--auto` can be added to the
subcommands (`build`, `flash, and `run`) and if the required options where
already passed and are recorded in the JSON file then TVMC would pick up and use
them automatically.

2. Similarly it's possible to remove the platform selection for other
subcommands after the platform is specified once in `create-project`. In that
case the project options would be discovered automatically based on the project
dir, not based on the template dir, as it's done in the current implementation.

# Future possibilities
[future-possibilities]: #future-possibilities

1. The current implementation and RFC does not address `tvmc tune` for micro
targets. It will be addressed later, once the interface here proposed is
discussed and crystallizes. The 'tvmc tune' will be extended probably along
the lines as the `tvmc run` command is extended.

2. For creating a project one departs from a compiled model kept in a Model
Library Format archive. This is fine and gives to the user total control over
which settings would be used when compiling a model (using `tvmc compile`). On
the other hand the CPU model for the target board, for instance, needs to be
specified explicitly, and that maybe annoying for some users. Thus it would be
good in the future if `create-project` could depart (pass as input instead of a
MLF archive) from a model not compiled (i.e. supply a model in one of the
frameworks’ format like TFLite, ONNX, etc) and TVMC would deduce a default
working set of options to compile the model based on the specified board. In
that case TVMC would need to keep a table to convert from the specified board to
the board's suitable default compile settings to be used for compiling the model
for the board. No change would be necessary for `tvmc compile` and
`tvmc create-project` would use it under the hood.
