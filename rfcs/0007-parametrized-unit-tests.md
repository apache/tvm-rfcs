- Feature Name: Parametrized Unit Tests
- Start Date: 2021-05-10(fill me in with today's date, YYYY-MM-DD)
- RFC PR: [apache/tvm-rfcs#0007](https://github.com/apache/tvm-rfcs/pull/0007)
- GitHub PR: [apache/tvm#8010](https://github.com/apache/tvm/issues/8010)

# Summary
[summary]: #summary

This RFC documents how to implement unit tests that depend on input
parameters, or have setup that depends on input parameters.

# Motivation
[motivation]: #motivation

Some unit tests should be tested along a variety of parameters for
better coverage.  For example, a unit test that does not depend on
target-specific features should be tested on all targets that the test
platform supports.  Alternatively, a unit test may need to pass
different array sizes to a function, in order to exercise different
code paths within that function.

The simplest implementation would be to write a test function that
loops over all parameters, throwing an exception if any parameter
fails the test.  However, this does not give full information to a
developer, as a failure from any parameter results in the entire test
to be marked as failing.  A unit-test that fails for all targets
requires different debugging than a unit-test that fails on a single
specific target, and so this information should be exposed.

This RFC adds functionality for implementing parameterized unit tests,
such that each set of parameters appears as a separate test result in
the final output.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

## Parameters

To make a new parameter for unit tests to use, define it with the
`tvm.testing.parameter` function.  For example, the following will
define a parameter named `array_size` that has three possible values.
This can appear either at global scope inside a test module to be
usable by all test functions in that module, or in a directory's
`conftest.py` to be usable by all tests in that directory.

```python
array_size = tvm.testing.parameter(8, 256, 1024)
```

To use a parameter, define a test function that accepts the parameter
as an input.  This test will be run once for each value of the
parameter.  For example, the `test_function` below would be run three
times, each time with a different value of `array_size` according to
the earlier definition.  These would show up in the output report as
`test_function[8]`, `test_function[256]`, and `test_function[1024]`,
with the name of the parameter as part of the function.

```python
def test_function(array_size):
    input_array = np.random.uniform(size=array_size)
    # Test code here
```

If a parameter is used by a test function, but isn't declared as a
function argument, it will produce a `NameError` when accessed.  This
happens even if the parameter is defined at module scope, and would
otherwise be accessible by the usual scoping rules.  This is
intentional, as access of the global variable would otherwise access
an `array_size` function definition, rather than the specific
parameter value.

```python
def test_function_broken():
    # Throws NameError, undefined variable "array_size"
    input_array = np.random.uniform(size=array_size)
    # Test code here
```

By default, a test function that accepts multiple parameters as
arguments will be run for all combinations of values of those
parameters.  If only some combinations of parameters should be used,
the `tvm.testing.parameters` function can be used to simultaneously
define multiple parameters.  A test function that accepts parameters
that were defined through `tvm.testing.parameters` will only be called
once for each set of parameters.

```python
array_size = tvm.testing.parameter(8, 256, 1024)
dtype = tvm.testing.parameter('float32', 'int32')

# Called 6 times, once for each combination of array_size and dtype.
def test_function1(array_size, dtype):
    assert(True)

test_data, reference_result = tvm.testing.parameters(
    ('test_data_1.dat', 'result_1.txt'),
    ('test_data_2.dat', 'result_2.txt'),
    ('test_data_3.dat', 'result_3.txt'),
)

# Called 3 times, once for each (test_data, reference_result) tuple.
def test_function3(test_data, reference_result):
    assert(True)
```

## Fixtures

Fixtures in pytest separate setup code from test code, and are used
for two primary purposes.  The first is for improved readability when
debugging, so that a failure in the setup is distinguishable from a
failure in the test.  The second is to avoid performing expensive test
setup that is shared across multiple tests, letting the test suite run
faster.

For example, the following function first reads test data, and then
performs tests that use the test data.

```python
# test_function_old() calls read_test_data().  If read_test_data()
# throws an error, test_function_old() shows as a failed test.

def test_function_old():
    dataset = read_test_data()
    assert(True) # Test succeeds
```

This can be pulled out into a separate setup function, which the test
function then accepts as an argument.  In this usage, this is
equivalent to using a bare `@pytest.fixture` decorator.  By default,
the fixture value is recalculated for every test function, to minimize
the potential for interaction between unit tests.

```python
@tvm.testing.fixture
def dataset():
    print('Prints once for each test function that uses dataset.')
    return read_test_data()

# test_function_new() accepts the dataset fixture.  If
# read_test_data() throws an error, test_function_new() shows
# as unrunnable.
def test_function_new(dataset):
    assert(True) # Test succeeds
```

If the fixture is more expensive to calculate, then it may be worth
caching the computed fixture.  This is done with the
`cache_return_value=True` argument.

```python
@tvm.testing.fixture(cache_return_value = True)
def dataset():
    print('Prints once no matter how many test functions use dataset.')
    return download_test_data()

def test_function(dataset):
    assert(True) # Test succeeds
```

The caching can be disabled entirely by setting the environment
variable `TVM_TEST_DISABLE_CACHE` to a non-zero integer.  This can be
useful to re-run tests that failed, to check whether the failure is
due to modification/re-use of a cached value.

A fixture can depend on parameters, or on other fixtures.  This is
defined by accepting additional parameters.  For example, consider the
following test function.  In this example, the calculation of
`correct_output` depends on the test data, and the `schedule` depends
on some block size.  The `generate_output` function contains the
functionality to be tested.

```python
def test_function_old():
    dataset = download_test_data()
    correct_output = calculate_correct_output(dataset)
    for block_size in [8, 256, 1024]:
        schedule = setup_schedule(block_size)
        output = generate_output(dataset, schedule)
        tvm.testing.assert_allclose(output, correct_output)
```

These can be split out into separate parameters and fixtures to
isolate the functionality to be tested.  Whether to split out the
setup code, and whether to cache it is dependent on the test function,
how expensive the setup is to perform the setup, whether other tests
can share the same setup code, and so on.

```python
@tvm.testing.fixture(cache_return_value = True)
def dataset():
    return download_test_data()
    
@tvm.testing.fixture
def correct_output(dataset):
    return calculate_correct_output(dataset)
    
array_size = tvm.testing.parameter(8, 256, 1024)

@tvm.testing.fixture
def schedule(array_size):
    return setup_schedule(array_size)
    
def test_function_new(dataset, correct_output, schedule):
    output = generate_output(dataset, schedule)
    tvm.testing.assert_allclose(output, correct_output)
```

## Target/Device Parametrization

The TVM test configuration contains definitions for `target` and
`dev`, which can be accepted as input by any test function.  These
replace the previous use of `tvm.testing.enabled_targets()`.

```python
def test_function_old():
    for target, dev in tvm.testing.enabled_targets():
        assert(True) # Target-based test
        
def test_function_new(target, dev):
    assert(True) # Target-based test
```

The parametrized values of `target` are read from the environment
variable `TVM_TEST_TARGETS`, a semicolon-separated list of targets.
If `TVM_TEST_TARGETS` is not defined, the target list falls back to
`tvm.testing.DEFAULT_TEST_TARGETS`.  All parametrized targets have
appropriate markers for checking device capability
(e.g. `@tvm.testing.uses_gpu`).  If a platform cannot run a test, it
is explicitly listed as being skipped.

It is expected both that enabling unit tests across additional targets
may uncover several unit tests failures, and that some unit tests may
fail during the early implementation of supporting a new runtime or
hardware.  In these cases, the `@tvm.testing.known_failing_targets`
decorator can be used.  This marks a test with `pytest.xfail`,
allowing the test suite to pass.  This is intended for cases where an
implementation will be improved in the future.

```python
@tvm.testing.known_failing_targets("my_newly_implemented_target")
def test_function(target, dev):
    # Test fails on new target, but marking as xfail allows CI suite
    # to pass during development.
    assert(target != "my_newly_implemented_target")
```

If a test should be run over a most targets, but isn't applicable for
some particular targets, the test should be marked with
`@tvm.testing.exclude_targets`.  For example, a test that exercises
GPU capabilities may wish to be run against all targets except for
`llvm`.

```python
@tvm.testing.excluded_targets("llvm")
def test_gpu_functionality(target, dev):
    # Test isn't run on llvm, is excluded from the report entirely.
    assert(target != "llvm")
```

If a testing should be run over only a specific set of targets and
devices, the `@tvm.testing.parametrize_targets` decorator can be used.
It is intended for use where a test is applicable only to a specific
target, and is inapplicable to any others (e.g. verifying
target-specific assembly code matches known assembly code).  In most
circumstances, `@tvm.testing.exclude_targets` or
`@tvm.testing.known_failing_targets` should be used instead.  For
example, a test that verifies vulkan-specific code generation should
be marked with `@tvm.testing.parametrize_targets("vulkan")`.

```python
@tvm.testing.parametrize_targets("vulkan")
def test_vulkan_codegen(target):
    f = tvm.build(..., target)
    assembly = f.imported_modules[0].get_source()
    assert("%v4bool = OpTypeVector %bool 4" in assembly)
```

The bare decorator `@tvm.testing.parametrize_targets` is maintained
for backwards compatibility, but is no longer the preferred style.

## Running Test Subsets

Individual python test files are no longer executable outside of the
pytest framework.  To maintain the existing behavior of running the
tests defined in a particular file, the following change should be
made.

```python
# Before
if __name__=='__main__':
    test_function_1()
    test_function_2()
    ...
    
# After
if __name__=='__main__':
    sys.exit(pytest.main(sys.argv))
```

Alternatively, single files, single tests, or single parameterizations
of tests can be explicitly specified when calling pytest.

```bash
# Run all tests in a file
python3 -mpytest path_to_my_test_file.py

# Run all parameterizations of a single test
python3 -mpytest path_to_my_test_file.py::test_function_name

# Run a single parameterization of a single test.  The brackets should
# contain the parameters as listed in the pytest verbose output.
python3 -mpytest 'path_to_my_test_file.py::test_function_name[1024]'
```


## Cache-Related Debugging

If test failure is suspected to be due to multiple tests having access
to the same cached value, the source of the cross-talk can be narrowed
down with the following steps.

1. Test with `TVM_TEST_DISABLE_CACHE=1`.  If the error stops, then the
   issue is due to some cache-related cross-talk.
    
2. Reduce the number of parameters being used for a single unit test,
   overriding the global parameter definition by marking it with
   `@pytest.mark.parametrize`.  If the error stops, then the issue is
   due to cross-talk between different parametrizations of a single
   test.
   
3. Run a single test function using `python3 -mpytest
   path/to/my/test_file.py::test_my_test_case`.  If the error stops,
   then the issue is due to cross-talk between the failing unit test
   and some other unit test in the same file.
   
   1. If it is due to cross-talk between multiple unit tests, run the
      failing unit test alongside each other unit test in the same
      file that makes use of the cached fixture.  This is the same
      command-line as above, but passing multiple test cases as
      arguments.  If the error stops when run with a particular unit
      test, then that test is the one that is modifying the cached
      fixture.
   
4. Run a single test function on its own, with a single
   parametrization, using `python3 -mpytest
   path/to/my/test_file.py::test_my_test_case[parameter_value]`.  If
   the error still occurs, and is still avoided by using
   `TVM_TEST_DISABLE_CACHE=1`, then the error is in
   `tvm.testing._fixture_cache`.


# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Both `tvm.testing.parameter` and `tvm.testing.fixture` are implemented
on top of `pytest.fixture`.  A call to `tvm.testing.parameter` defines
a fixture that takes specific values.  The following two definitions
of `array_size` are equivalent.

```python
# With new functionality
array_size = tvm.testing.parameter(8, 256, 1024)

# With vanilla pytest functionality
@pytest.fixture(params=[8, 256, 1024])
def array_size(request):
    return request.param
```

The `@tvm.testing.fixture` without any arguments is equivalent to the
`@pytest.fixture` without any arguments.

```python
@tvm.testing.fixture
def test_data(array_size):
    return np.random.uniform(size=array_size)
    
@pytest.fixture
def test_data(array_size):
    return np.random.uniform(size=array_size)
```

The `@tvm.testing.fixture(cached_return_value=True)` does not have a
direct analog in vanilla pytest.  While pytest does allow for re-use
of fixtures between functions, it only ever maintains [a single cached
value of each
fixture](https://docs.pytest.org/en/6.2.x/fixture.html#fixture-scopes).
This works in cases where only a single cached value is required, but
causes repeated calls to setup code if a test requires multiple
different cached values.  This can be reduced by careful ordering of
the pytest fixture scopes, but cannot be completely eliminated.  The
different possible cache usage in vanilla pytest, and with
`tvm.testing.fixture` are shown below.

```python
# Possible ordering of tests if `target` is defined in a tighter scope
# than `array_size`.  The call to `generate_setup2` is repeated.
for array_size in array_sizes:
    setup1 = generate_setup1(array_size)
    for target in targets:
        setup2 = generate_setup2(target)
        run_test(setup1, setup2)
        
# Possible ordering of tests if `target` is defined in a tighter scope
# than `array_size`.  The call to `generate_setup2` is repeated. 
for target in targets:
    setup2 = generate_setup2(target)
    for array_size in array_sizes:
        setup1 = generate_setup1(array_size)
        run_test(setup1, setup2)
        
# Pseudo-code equivalent of `tvm.testing.fixture(cache_return_value=True)`.  
# No repeated calls to setup code.
cache_setup1 = {}
cache_setup2 = {}
for array_size in array_sizes:
    for target in targets:
        if array_size in cache_setup1:
            setup1 = cache_setup1[array_size]
        else:
            setup1 = cache_setup1[array] = generate_setup1(array_size)

        if target in cache_setup2:
            setup2 = cache_setup2[target]
        else:
            setup2 = cache_setup2[target] = generate_setup2(target)

        run_test(setup1, setup2)

del cache_setup1
del cache_setup2
```

The cache for a fixture defined with `tvm.testing.fixture` is cleared
after all tests using that fixture are completed, to avoid excessive
memory usage.

If a test function is marked with `@pytest.mark.parametrize` for a
parameter that is also defined with `tvm.testing.parameter`, the test
function uses only the parameters in `@pytest.mark.parametrize`.  This
allows an individual function to override the parameter definitions if
needed.  Any parameter-dependent fixture are also determined based on
the values in `@pytest.mark.parametrize`.

# Drawbacks
[drawbacks]: #drawbacks

- This makes the individual unit tests be more dependent on the test
  framework and setup.  Incorrect setup may result in confusing test
  results.

- Caching setup between different tests introduces potential
  cross-talk between tests.  While this risk is also present when
  looping over parameter values, separating cached values out into
  fixtures hides that potential cross-talk.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

- Option: Explicitly loop over parameter values or
  `tvm.testing.enabled_parameters` in the test function.  (Most common
  previous usage.)
  
  - Pros:
    - Explicit at the definition of a test function.
    
  - Cons:
    - Requires opt-in at each test functions.
    - Doesn't report information on which parameter value(s) failed.
    
    
- Option: Use `@tvm.testing.parametrize_targets` as a bare fixture.
  (Previously implemented behavior, less common usage.)

  - Pros:
    - Explicit at the definition of a test function.
    
  - Cons:
    - Requires opt-in at each test function.
    - Doesn't provide functionality for shared setup.
    

- Option: Pararametrize using `@pytest.mark.parametrize` rather than
  `tvm.testing.parameter`.
  
  - Pros:
    - Would explicitly show the parameter values next to the function
      it applies to.
      
  - Cons:
    - Must be explicitly added at each test function definition.
    - Extending the parameters that apply across all tests in a
      file/directory requires updating several locations.
    - Similar parameters (e.g. 1000 vs 1024 for an array length) would
      be defined at separate locations, and would then require
      separate fixture setup.

# Prior art
[prior-art]: #prior-art

Discuss prior art, both the good and the bad, in relation to this proposal.
A few examples of what this can include are:

- Does this feature exist in other ML compilers or languages and discuss the experince their community has had?
- For community proposals: Is this done by some other community and what were their experiences with it?
- For other teams: What lessons can we learn from what other communities have done here?
- Papers: Are there any published papers or great posts that discuss this? 
  If you have some relevant papers to refer to, this can serve as a more detailed theoretical background.

If there is no prior art, that is fine - your ideas are interesting to us whether they are 
  brand new or if it is an adaptation from other languages.

Note that while precedent set by other languages is some motivation, it does not on its own motivate an RFC.
Please also take into consideration that TVM intentionally diverges from other compilers.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

- What values are appropriate to cache using
  `@tvm.testing.fixture(cache_return_value=True)`?  Should
  non-serializable values be allowed?

  If only serializable values are allowed to be cached, this may aid
  in debugging, since the values of all test parameters and cached
  fixtures could be saved and reproduced.

  Currently, nearly all cases (e.g. datasets, array sizes, targets)
  are serializable.  The only non-serializable case after
  brainstorming would be RPC server connections.  There is some
  concern that caching RPC server connections could cause difficulties
  in reproducing test failures.

  Current proposed answer is to only cache serializable values, and
  that the discussion can be resumed when we have other possible use
  cases for caching non-serializable values.

# Future possibilities
[future-possibilities]: #future-possibilities

- Parameters common across many tests could be defined at a
  larger scope (e.g. `${TVM_HOME}/conftest.py`) and be usable in a
  file without additional declaration.
  
- Parameters common across many tests could have additional randomly
  generated values added to the list, adding fuzzing to the tests.
  
- Parametrized unit tests interact very nicely with the
  [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/stable/)
  plugin for comparing low-level functionality.  For example, the
  definition below would benchmark and record statistics for the
  runtime to copy data from a device to the CPU, with the benchmarks
  tagged by the parameter values of `array_size`, `dtype`, and
  `target`.  The benchmarking can be disabled by default and run only
  with the `--benchmark-enable` command-line argument.

  ```python
  def test_copy_data_from_device(benchmark, array_size, dtype, dev):
      A = tvm.te.placeholder((array_size,), name="A", dtype=dtype)
      a_np = np.random.uniform(size=(array_size,)).astype(A.dtype)
      a = tvm.nd.array(a_np, dev)
  
      b_np = benchmark(a.numpy)
      tvm.testing.assert_allclose(a_np, b_np)
  ```
