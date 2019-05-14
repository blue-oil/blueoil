IMAGE_NAME:=blueoil_$$(id -un)
BUILD_VERSION:=$(shell git describe --tags --always --dirty --match="v*" 2> /dev/null || cat $(CURDIR/.version 2> /dev/null || echo v0))
DOCKER_OPT:=--runtime=nvidia

default: build

.PHONY: deps
deps:
	# Update dependencies
	git submodule update --init --recursive

.PHONY: build
build: deps
	# Build docker image
	docker build -t $(IMAGE_NAME):$(BUILD_VERSION) --build-arg python_version="3.6.3" -f docker/Dockerfile .

.PHONY: test
test: build
	# Run Blueoil test
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) bash ./blueoil_test.sh

.PHONY: test-lmnet
test-lmnet: build
	# Run lmnet test with Python3.6 (only available on Jenkins)
	docker run $(DOCKER_OPT) -v /storage/dataset:/storage/dataset -v lmnet_saved:/storage/lmnet/saved -e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) -e DATA_DIR=/storage/dataset --rm $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cd lmnet; tox -e py36"

.PHONY: test-dlk
test-dlk: build
	# Run dlk test
	docker run --rm -t -v $(HOME)/.ssh:/tmp/.ssh -e FPGA_HOST --net=host $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cp -R /tmp/.ssh /root/.ssh && apt-get update && apt-get install -y iputils-ping && cd dlk && python setup.py test"

.PHONY: pep8-dlk
pep8-dlk: build
	# Check dlk PEP8
	docker run --rm -t $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cd dlk && pycodestyle --ignore=W --max-line-length=120 --exclude='*static/pb*','*docs/*','*.eggs*','*tvm/*','*tests/*','backends/*' ."

.PHONY: clean
clean:
	# Clean created files
	docker rmi  $(IMAGE_NAME):$(BUILD_VERSION)
	rm -rf tmp/*
