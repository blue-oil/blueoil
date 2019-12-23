#!/bin/bash
set -euf

# test cases to run inference
# <buildkite agent-type> <model filename>
TEST_CASES=(
    "de10nano       lib/lib_arm.so"
    "de10nano       lib/lib_fpga.so"

    # "ultra96        lib/lib_aarch64.so"
    # "ultra96        lib/lib_fpga.so"

    "jetson-nano    lib/lib_aarch64.so"
    "jetson-nano    minimal_graph_with_shape.pb"

    "jetson-tx2     lib/lib_aarch64.so"
    "jetson-tx2     minimal_graph_with_shape.pb"

    "jetson-xavier  lib/lib_aarch64.so"
    "jetson-xavier  minimal_graph_with_shape.pb"

    # "lm-server      lib/lib_x86.so"
    # "lm-server      minimal_graph_with_shape.pb"

    "raspberry-pi   lib/lib_aarch64.so"
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
    read AGENT MODEL <<< "${TEST_CASE}"

    if [ "${AGENT}" = "de10nano" ]; then
        PYTHON_COMMAND="sudo python"
    else
        PYTHON_COMMAND="python"
    fi

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
      ${PYTHON_COMMAND} run.py -i ../../inference_test_data/raw_image.png -c ../models/meta.yaml -m ../models/${MODEL}

    artifact_paths:
      - "export/*/*/output/python/output/output.json"


EOS
done
