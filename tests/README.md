# CI Pipelines for Blueoil
We have several CI Pipelines
* Buildkite Pipelines : https://buildkite.com/blueoil

## Automated Test with PR
The tests run automatically by making PR.

The triggers to run tests are below.
* Create PR
* Push commit to branch related PR
* Comment in PR
    * Run all tests
        * Comment `run test`
    * Run the specified test individually
        * blueoil test : `run blueoil test`
        * lmnet test : `run lmnet test`
        * dlk test : `run dlk test`
* Add label in PR
    * Run all tests
        * Add `CI: test-all`
    * Run the specified test individually
        * blueoil test : Add `CI: test-blueoil`
        * lmnet test : Add `CI: test-lmnet`
        * dlk test : Add `CI: test-dlk`

## Test Jobs
All `make` tasks should be made in repository root dir.

### [blueoil] test blueoil entire workflow with GPU
#### Buildkite
See: [.buildkite/blueoil-pipeline.yml](../.buildkite/blueoil-pipeline.yml)

### [lmnet] test lmnet training part with GPU
#### Buildkite
See: [.buildkite/lmnet-pipeline.yml](../.buildkite/lmnet-pipeline.yml)

### [dlk] test compiling and inference part on FPGA device
#### Buildkite
See: [.buildkite/dlk-pipeline.yml](../.buildkite/dlk-pipeline.yml)
