# This Makefile is only for documentation

.PHONY: python cpp clean

all: cpp python

python:
	cd python; PYTHONPATH=../../dlk/python/dlk make html

cpp:
	cd cpp; doxygen

clean:
	cd python; make clean; \
	cd ../cpp; rm -rf html latex
