CHISEL_FILES := $(wildcard ../../src/main/scala/**/*.scala)

.PHONY: bxb_ip
bxb_ip: bxb_ip/Bxb.v

bxb_ip/Bxb.v: $(CHISEL_FILES)
	cd ../../ && sbt "runMain bxb.Bxb --output-file $(abspath $@)"
