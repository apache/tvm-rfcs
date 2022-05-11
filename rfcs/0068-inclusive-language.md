- Feature Name: Inclusive Language
- Start Date: 2022-05-04
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/0068)
- GitHub Issue: [apache/tvm-rfcs#68](https://github.com/apache/tvm-rfcs/pull/68)

# Summary
[summary]: #summary

This RFC proposes to remove some non-inclusive language from the existing TVM codebase and documentation. It also proposes to introduce a linting script to CI that will prevent these non-inclusive terms from being re-introduced.

# Motivation
[motivation]: #motivation

In order for TVM to be an open and inclusive community it is important that we are mindful of the language we use in our code and documentation. In particular, we should try where possible to avoid using terminology that other contributors may find offensive.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

In TVM we try where possible to avoid using the following terms:
* Master and Slave
* Blacklist and Whitelist
* White Box
* The use of gender-specific pronouns when gender is either irrelevant or unknown

Some suggested replacements are:
* Main in place of Master
* Requester/Completer in place of Master/Slave
* Allowlist or Allowed *Noun* in place of Whitelist
* DenyList or Denied *Noun* in place of Blacklist
* It, They, Them, Theirs in place of He/She, Him/Her and His/Hers

In order to prevent non-inclusive language from inadvertently being added to the codebase or documentation, a linting script is run on the CI system for each PR and use of such terms will result in a CI failure.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

The first part of the proposal is to manually replace any existing use of non-inclusive terms in the codebase and documentation. 

It will attempt to replace the following non-inclusive terms with suitable replacements:
* Master and Slave
* Blacklist and Whitelist
* White Box
* The use of gender-specific pronouns when gender is either irrelevant or unknown

We acknowledge that it will not be possible to replace instances of non-inclusive terms in every single case. In particular, where there is a dependency on a third party tool or library that uses such terms, it may not be possible to make a replacement. This is the case for many URLs in the existing codebase that include the word "master", since they exist on the "master" branch in some repository. In this case these terms will be left as they are.

The second part of the proposal involves using the [blockint](https://github.com/PrincetonUniversity/blocklint) command line utility for finding non-inclusive terms.

The blocklint command line utility will be installed in the ci_lint docker image.

A CI script will be created that passes the following terms to blocklint in the blocklist parameter:
* Master
* Slave
* Blacklist
* Whitelist
* White Box

It doesn't seem practical to pass gender specific pronouns in the blocklist since these seem to generate many false positives. Instead we propose that the use of gender-specific pronouns when gender is irrelevant or unknown be monitored during code reviews.

Since it is not possible to always replace non-inclusive terms e.g. in the case of references to third party tools, a list of files to skip during linting will be passed in the skiplist parameter to blocklint. In addition, blocklint can be instructed to ignore a particular occurrence of a blocked word, by including a comment "blocklint: pragma" on the same line. Some examples of this use can be seen in the [pragma tests for blocklint](https://github.com/PrincetonUniversity/blocklint/blob/master/tests/sample_files/test_pragma.cc).

# Drawbacks
[drawbacks]: #drawbacks

* There is the possibility of resistance to the replacement of certain terms by the community.
* The linting script may well generate false positives which need to be excluded by adding to the skiplist.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

* In terms of replacing existing non-inclusive language in the codebase, it seems self-evident that this would be the right thing to do. The alternative is to leave these terms in place and risk alienating other members or potential contributors to the TVM community.
* In terms of linting tools, the [woke](https://github.com/get-woke/woke) command line tool was also considered, but it has very similar functionality to blocklint, and there didn't seem to be a compelling reason to select it over [blockint](https://github.com/PrincetonUniversity/blocklint).

# Prior art
[prior-art]: #prior-art

The following open source projects have implemented broadly similar initiatives w.r.t. inclusive language:

* Red Hat: [Making open source more inclusive by eradicating problematic language](https://www.redhat.com/en/blog/making-open-source-more-inclusive-eradicating-problematic-language)
* Google: [Google’s initiative for more inclusive language in open source projects](https://opensource.googleblog.com/2020/11/googles-initiative-for-more-inclusive.html)
* python: [Avoid master/slave terminology](https://github.com/python/cpython/issues/78786)
* github: [Github plans to replace racially insensitive terms like ‘master’ and ‘whitelist’](https://thenextweb.com/news/github-plans-to-replace-racially-insensitive-terms-like-master-and-whitelist)


# Future possibilities
[future-possibilities]: #future-possibilities

* The list of non-inclusive terms could be modified/expanded on the basis of input from the TVM community.
* There are possibilities for improvement to the blocklint tool itself.
  * It would be useful to be able to pass a list of "exact match" words to ignore. For example, being able ignore the term "/master/" (i.e. when used in a URL) would avoid false positives on many references to third party URLs that can't be changed.
