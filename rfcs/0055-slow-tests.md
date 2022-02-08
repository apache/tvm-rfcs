- Feature Name: `slow_tests_decorator`
- Start Date: 2022-01-26
- RFC PR: [apache/tvm-rfcs#55](https://github.com/apache/tvm-rfcs/pull/55)

# Summary

[summary]: #summary

Add a Python decorator to skip tests on PRs but run them on branches (e.g. `main`).

# Motivation

[motivation]: #motivation

A small subset of tests take up a large portion of the total test runtime in Pull Requests (PRs). This
RFC proposes that we skip these tests on PRs where CI runtime is critical and
run them only on `main`.

# Guide-level explanation

[guide-level-explanation]: #guide-level-explanation

CI runtime constantly plagues TVM developers with long iteration times, exacerbated
by flakiness and difficult-to-reproduce steps. To reduce runtime, we can execute
more work concurrently and increase usage of available resources, usually by
parallelization within ([apache/tvm#9834](https://github.com/apache/tvm/pull/9834))
or between ([apache/tvm#9733](https://github.com/apache/tvm/pull/9733)) CI jobs.
Another way is to do less work, which is what this RFC proposes. By running some
tests on `main` only, we still get some measure of coverage provided by these tests
without burdening PR developers.

The runtime savings of this change are potentially significant, as this gives us
a black-box knob which we can manually tune over time to trade off between PR test
coverage and PR test runtime. [This gist](https://gist.github.com/driazati/e009f09ff44c6bc91c4d95a8e17fd6f1)
(see the "Details" below for a sample) shows a listing of TVM test runtime in descending
order, showing the potential time savings (across all jobs) in CI of cutting
off tests above an arbitrary runtime (not to propose use a cutoff, but just to
demonstrate in broad strokes the potential impact of this change):

```
[cutoff=10.0s] Total savings (m): 419.95m by skipping 695 tests
[cutoff=20.0s] Total savings (m): 338.27m by skipping 320 tests
[cutoff=30.0s] Total savings (m): 291.5m by skipping 205 tests
[cutoff=40.0s] Total savings (m): 251.59m by skipping 135 tests
[cutoff=50.0s] Total savings (m): 222.56m by skipping 96 tests
[cutoff=60.0s] Total savings (m): 203.3m by skipping 75 tests
[cutoff=70.0s] Total savings (m): 192.68m by skipping 65 tests
[cutoff=80.0s] Total savings (m): 181.56m by skipping 56 tests
[cutoff=90.0s] Total savings (m): 171.45m by skipping 49 tests
[cutoff=100.0s] Total savings (m): 160.36m by skipping 42 tests
```

<details>

Top 20 slowest tests of https://gist.github.com/driazati/e009f09ff44c6bc91c4d95a8e17fd6f1

```
runtime (s)	file	test
1044.31	tests/python/frontend/tensorflow/test_forward.py	test_forward_broadcast_args
697.41	tests/python/frontend/tensorflow/test_forward.py	test_forward_broadcast_to
624.77	tests/python/frontend/tensorflow/test_forward.py	test_forward_ssd
567.74	tests/python/frontend/tflite/test_forward.py	test_all_elemwise
433.44	tests/python/topi/python/test_topi_upsampling.py	test_upsampling3d
329.4	tests/python/topi/python/test_topi_conv2d_int8.py	test_conv2d_nchw
326.02	tests/python/frontend/pytorch/test_object_detection.py	test_detection_models
282.74	tests/python/frontend/tflite/test_forward.py	test_forward_transpose_conv
280.26	tests/python/topi/python/test_topi_conv2d_hwnc_tensorcore.py	test_conv2d_hwnc_tensorcore
277.15	tests/python/topi/python/test_topi_conv3d_transpose_ncdhw.py	test_conv3d_transpose_ncdhw
249.39	tests/python/topi/python/test_topi_conv2d_NCHWc.py	test_conv2d_NCHWc
243.81	tests/python/relay/test_py_converter.py	test_global_recursion
227.9	tests/python/frontend/pytorch/test_forward.py	test_segmentation_models
194.23	tests/python/relay/test_op_level6.py	test_topk
183.41	tests/python/frontend/tensorflow/test_forward.py	test_forward_ptb
178.62	tests/python/relay/test_py_converter.py	test_global_recursion
171.25	tests/python/frontend/pytorch/qnn_test.py	test_quantized_imagenet
169.2	tests/python/frontend/tensorflow/test_forward.py	test_forward_resnetv2
169.13	tests/python/topi/python/test_topi_conv2d_int8.py	test_conv2d_nhwc
```

</details>

Usages of `@slow` will first be limited to manually inspected tests that have
low flakiness and infrequent failures on PRs in general in order to have the
least impact on test coverage on PRs. The relevant test owners will also need
to approve of their tests being moved to `main` at first with `@slow`.

# Reference-level explanation

[reference-level-explanation]: #reference-level-explanation

A decorator `@tvm.testing.slow` will be added (see [apache/tvm#10057](https://github.com/apache/tvm/pull/10057)) that implements
the above behavior. Skipping slow tests would be an opt-in, rather than opt-out.
This way developers who don't read this RFC won't have to adjust their workflows
at all to run these tests locally. There is also a need to run slow tests on PRs
in some cases, such as fixing reverted commits or if a developer suspects their
change would have a wide reaching impact. In this case, `@ci run slow tests` can
be added to the PR body before tests are run in order to disable skipping slow tests.
A similar mechanism could be implemented in C++ using [`GTEST_SKIP`](https://github.com/google/googletest/blob/main/docs/advanced.md#skipping-test-execution).

Using a decorator has the advantage of being explicit compared to an automated system
to detect and skip slow tests. Clicking through to the decorator’s short definition
makes it clear what is going on so this shouldn’t add too much of a development burden.
Slow tests run by default to minimize disruption but can be controlled by setting
`SKIP_SLOW_TESTS=1`, which would affect all slow test filtering fixtures (in C++ and Python).

An environment variable `SKIP_SLOW_TESTS=1` will be set in Jenkins on PRs. Branches,
including `main` and `ci-docker-staging` will not have this flag set and will
always run the full set of tests.

# Drawbacks

[drawbacks]: #drawbacks

The primary caveat is that tests that run on `main` may now fail due to PRs that
were green when they were merged, so this will require some buy-in from all TVM
developers. However, the runtime savings are significant (see below) enough to make
this worth it. Developments like [apache/tvm#9554](https://github.com/apache/tvm/pull/9554) will make the revert process much
smoother as well to minimize disruptions.

# Rationale and alternatives

[rationale-and-alternatives]: #rationale-and-alternatives

This isn't a complete solution. Most PRs end up running lots of tests that the
PR didn't affect at all. Ideally we would have to determine the dependency graph
of PRs based solely on files changed, but this isn't generally possible without
restricting Python and making big changes to the existing TVM build system.
[testmon](https://testmon.org/) is another approach, which uses coverage data to
determine which tests to run, though Python also makes this difficult to implement correctly.
This could also be implemented at the human level, with developers tagging their
PRs based on what they think should run, though this has a higher potential to
miss certain tests. However, this run-what-changed future would be difficult to
achieve.

Other efforts involve looking into tests themselves to determine why they are slow.
Often TVM's tests are running much more work than they actually intend to test
(such as using entire off-the-shelf networks to test a few operators) in
more of an integration test than a unit test. Replacing these types of test with
a framework that makes it easier to test TVM passes and functionality in smaller
chunks is related but orthogonal to this work, requiring significantly higher
resources due to the need to implement testing infrastructure for passes and
inspect the relevant tests (though it is on our roadmap in the near future).
Over time as slow tests are manually debugged, `@slow` decorators could be removed.

# Prior art

[prior-art]: #prior-art

- Tensorflow skips long tests: https://www.tensorflow.org/community/contribute/tests#test_times_should_aim_for_half_of_test_size_timeout_to_avoid_flakes
- PyTorch does essentially the same thing with a `@slow` decorator but maintains
  an automatically updating list of slow tests to skip:
  https://github.com/pytorch/pytorch/blob/master/torch/testing/_internal/common_utils.py#L748-L755
- scikit-learn has a similar system for low-signal, slow tests: https://github.com/scikit-learn/scikit-learn/pull/21645

# Unresolved questions

[unresolved-questions]: #unresolved-questions

- What tests do we `@slow`? Based on discussion it seems like going down the list of slow tests one by one (at least to start), it is prudent to do a preliminary investigation to answer:

  1. What is this test? Should it be slow?
  2. How often does this test fail? If the test fails often, there is less of a case that it should be `@slow`-ed since it provides good signal to developers.
  3. Who relies on this test? Do they understand the implications of `@slow`?

- Who will monitor `main` for PR-related breakages? What is the SLA on fixes? Recent additions such as [messaging Discord on `main` failures](https://github.com/tlc-pack/ci-monitoring) and keeping track of the last known good commit ([apache/tvm#10056](https://github.com/apache/tvm/pull/10056)) should make this easier.

# Future possibilities

[future-possibilities]: #future-possibilities

- Better communication in Jenkins job pages of which tests ran, which did not, and why
- Different levels of tests. `main` is the most frequent step, but longer running tests could be moved out to nightly or even release level testing (though this makes debugging failures more difficult).
- Gather test coverage data to get a basic idea of what code is being tested
- Track per-test durations for commits and report on CI runs what the runtime difference of each PR is (so developers can easily see the burden of their change)

