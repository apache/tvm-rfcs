- Feature Name: docker_image_tag_format
- Start Date: 2022-04-08
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary

Defines a format for TVM CI Docker images tag.

# Motivation

In the current format used to identify our Docker images, it is impossible to link back a given version of the image to which version of the repository that represents.

# Guide-level explanation

For years we have been using a pattern to version our CI Docker images, which is composed by v0.INCREMENTAL_NUMBER, e.g. tlcpack/ci_cpu:v0.80. Every time an image is updated, we bump the minor number, just as a way to release a new version.

As the project is growing and given we recently have been improving the Docker images building process, it is time to consider using a more meaningful tagging scheme, so that it is easier to identify what is included in the images being used in our CI.

When looking/using a Docker image, usually we would be interested in:

1. How long ago was this image generated?
2. What is the last change added in the current images being used in production?

None of these questions can be answered by our current numbering scheme, and to get that information we would usually need to do some digging and inspecting the image to see what is in there.

# Reference-level explanation

As an improvement to the current situation, this RFC proposes the adption of a tagging scheme currently used in the `tlcpackstaging` (https://hub.docker.com/u/tlcpackstaging) repository, composed by:

* a timestamp YYYYMMDD-HHMMSS
* the last short git hash added in that image

One example would be tlcpackstaging/ci_arm:20220201-115323-2af42ba8e. This tells us when this image was generated and up to which point in the repository is included.

The advantage with the proposed scheme, is that we can just look back on our own repository to discover which changes to the Docker setup scripts (under docker/install/* in the TVM repository) as a way to check whether a given change is expected to be in the image.

# Drawbacks

Pointing to "the previous image" is obvious today, because it is just "the current version minus one". To revert the image in the proposed tagging scheme, we will need to adapt this process and use git history to discover which was the previous version in the repository. However, reverting images is quite rare so it should be something the will consume lots of time very often.

# Rationale and alternatives

Many other formats could be used to describe the tag, such as `git describe`, however, it lacks the timestamp, which is a quite useful piece of information to give an impression of how long ago an image was updated.

# Prior art

The proposed format is used in our automated daily Docker image rebuild jobs at https://ci.tlcpack.ai/job/docker-images-ci/.

# Future possibilities

As this is already in discussion, there is an opportunity, given we evaluate properly the impact in the current CI job runs, to rebuild Docker images for each PR.
