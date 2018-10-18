# Contributing guidelines

Welcome to LMNet repository! We are glad to see that you might be interested in contributing to the project. But before doing that, below are the guidelines we would like you to follow.

* [Overview](#overview)
  * [Question or Problem?](#question-or-problem)
  * [Found a Bug?](#found-a-bug)
  * [Missing a Feature?](#missing-a-feature)
* [Pull Request Guidelines](#pull-request-guidelines)
  * [Coding Standard](#coding-standard)
  * [Type of pull requests](#types)
  * [Branch naming](#branch-naming)
  * [Commit message](#commit-message)


## Overview
Please follow this chart to either develop or make contribution.

![dev_flow.png](https://github.com/LeapMind/lmnet/blob/master/docs/dev_flow/dev_flow.png)

If you want to change the development flow, please see [README.md](https://github.com/LeapMind/lmnet/blob/master/docs/dev_flow/README.md)

### Question or Problem?
Please contact the contributors via slack first. If the problem is confirmed to be a bug or feature request, we would like you to open new issue.
Currently, we haven't setup any issue management hook yet. So please follow your own issue until it has been resolved or closed.

### Found a Bug?
If you find a bug in code, you can help us by opening an issue.
If the issue is valid, the contributors will create a task on Wrike and choose someone to work on it. If you can help us with a fix by submit a pull request is even better!

### Missing a Feature?
You can request a new feature by opening an issue. If you would like to implement it yourself, feel free!
If you struggle with any problem, just ask contributors for help. 

## Pull Request Guidelines

Before sending your pull request, there are several points that one should know.

* At least run your code once to make sure it works
* Add test for your code (see files under `tests` folder)
* Run tests (`pytest` and `flake8` currently)
* Once submitted, it will also run [Jenkins](#about-jenkins) automatically.
* Need reviewer

If you don't know how to do above things, see `README.md` and contact to contributor.

After sending your pull request, Please add labels.

### About Jenkins

* Jenkins is an automated server, located [here](https://jenkins.leapmind:8443).
* The user login/password (general purpose) is at [leapmind internal website](https://sites.google.com/a/leapmind.io/internal/home/infurachimu/jenkins?pli=1).
* Once inside, go to the job named **lmnet_pr_test**, find the #id of your run by looking at build history. Console output can be viewed for details about test passing/failure.
* If you need new files such as new datasets, please contact `infra team` to add related files into Jenkins.

### Coding Standard

* Please follow [PEP8](https://www.python.org/dev/peps/pep-0008).
* All public methods, such as quantizers, **must be documented**.
* Again, all features or bug fixes **must be tested**.

### Types

Type of issue, pull request, branch, commit must be one of the following:

* **docs**: Documentation-only changes
* **feat**: A new feature
* **fix**: A bug fix
* **perf**: A code change that improves performance
* **refactor**: A code change that neither fixes a bug nor adds a feature
* **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
* **test**: Adding missing tests or correcting existing tests
* **revert**: Reverts a previous commit

### Branch Naming

We prefer a descriptive branch name. Such as `feat_new_cls_network` to let others know what happens at this branch.

Please don't use `version` in the branch name. that is done with **`release`** tagging.

Sample:
```
fix_any_bug
feat_new_cls_network
test_add_new_qtz_test
```

### Commit Message
Sample:
```
fix(#issue_no_if_any): fix the bug

Fix the bug that ...
```

