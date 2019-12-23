#!/bin/bash
set -euf

# test cases to run inference
# <buildkite agent-type> <model filename>
TEST_CASES=(
    "de10nano      lib_arm.so"
    "de10nano      lib_fpga.so"
    # "ultra96        lib_aarch64.so"
    # "ultra96        lib_fpga.so"
    "jetson-nano    lib_aarch64.so"
    "jetson-tx2     lib_aarch64.so"
    "jetson-xavier  lib_aarch64.so"
    "raspberry-pi   lib_aarch64.so"
    # "lm-server      lib_x86.so"
)

cat <<EOS
steps:
  - label: "inference: setup"
    key: "setup"
    timeout_in_minutes: "30"
    agents:
      - "env=benchmark"
      - "agent-type=gcloudsdk"

    command: |
      gsutil -m cp "${CONVERT_RESULT_PATH}" ./

    artifact_paths:
      - "convert-result.tgz"


EOS

for TEST_CASE in "${TEST_CASES[@]}" ; do
    IFS=' '
    read AGENT MODEL <<< "$TEST_CASE"

    cat <<EOS
  - label: "inference: ${AGENT} (${MODEL})"
    depends_on: "setup"
    timeout_in_minutes: "30"

    agents:
      - "env=benchmark"
      - "agent-type=${AGENT}"

    command: |
      buildkite-agent artifact download "convert-result.tgz" ./
      tar xvf convert-result.tgz
      cd export/*/*/output/python
      pip install -r requirements.txt
      python run.py -i ../../inference_test_data/raw_image.png -c ../models/meta.yaml -m ../models/lib/${MODEL}

    artifact_paths:
      - "export/*/*/output/python/output/output.json"


EOS
done
