## Jenkins Server
Here is Jenkins server for CI with GPU: https://jenkins.blue-oil.org
Here is Jenkins server for CI with FPGA: https://jenkins.leapmind.local:8080 (only for local access)

## Auto Test with PR
This test will be run automatically by PR.

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


## Test Jobs
### [blueoil] jenkins test
* Cofiguration URL : https://jenkins.blue-oil.org/job/blueoil_main/configure
* Script run by Jenkins : `./.jenkins/test_blueoil.sh`

### [lmnet] jenkins test
* Cofiguration URL : https://jenkins.blue-oil.org/job/blueoil_lmnet/configure
* Script run by Jenkins : `./.jenkins/test_lmnet.sh`

### [dlk] jenkins test
* Cofiguration URL : http://jenkins.leapmind.local:8080/job/blueoil_dlk_test/configure
* Script run by Jenkins: `./.jenkins/test_dlk.sh`
