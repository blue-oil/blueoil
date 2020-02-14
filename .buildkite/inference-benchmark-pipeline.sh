#!/bin/bash
set -euf

# This script is executed by Buildkite to generate pipeline YAML string.
# Document: https://buildkite.com/docs/pipelines/defining-steps#dynamic-pipelines
#
# Usage: CONVERT_RESULT_PATH=gs://path/to/result.tgz ./inference-benchmark-pipeline.sh

# test cases to run inference
# "<buildkite agent-type> <model filename>"
TEST_CASES=(
    "de10nano       lib/libdlk_arm.so"
    "de10nano       lib/libdlk_fpga.so"

    "jetson-nano    minimal_graph_with_shape.pb"
    "jetson-tx2     minimal_graph_with_shape.pb"
    "jetson-xavier  minimal_graph_with_shape.pb"

    "raspberry-pi   lib/libdlk_aarch64.so"
)

# Generate setup step
cat <<EOS
steps:
  - label: "inference: setup"
    key: "setup"
    timeout_in_minutes: "30"
    agents:
      - "env=benchmark"
      - "agent-type=gcloudsdk"

    artifact_paths:
      - "convert-result.tgz"

    command: |
      gsutil -m cp "${CONVERT_RESULT_PATH}" ./
EOS

for TEST_CASE in "${TEST_CASES[@]}" ; do
    IFS=' '
    read AGENT MODEL <<< "${TEST_CASE}"

    # Generate inference step
    cat <<EOS


  - label: "inference: ${AGENT} (${MODEL})"
    depends_on: "setup"
    timeout_in_minutes: "30"

    agents:
      - "env=benchmark"
      - "agent-type=${AGENT}"

    artifact_paths:
      - "export/*/*/output/python/output/output.json"

    command: |
      buildkite-agent artifact download "convert-result.tgz" ./ --build \$\${BUILDKITE_BUILD_ID}
      tar xvf convert-result.tgz
      cd export/*/*/output/python
EOS

    # Generate commands to run inference (with and without "sudo")
    if [ "${AGENT}" = "de10nano" ] && [ "${MODEL}" = "lib/libdlk_fpga.so" ] ; then
        cat <<EOS
      sudo python run.py -i ../../inference_test_data/raw_image.png -c ../models/meta.yaml -m ../models/${MODEL}
      sudo chown -R buildkite-agent:buildkite-agent output
EOS
    else
        cat <<EOS
      python run.py -i ../../inference_test_data/raw_image.png -c ../models/meta.yaml -m ../models/${MODEL}
EOS
    fi
done
