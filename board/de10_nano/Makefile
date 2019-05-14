CHISEL_FILES := $(wildcard ../../src/main/scala/**/*.scala)

.PHONY: bxb_ip
bxb_ip: bxb_ip/Bxb.v

.PHONY: soc_system
soc_system: soc_system.done_stamp

bxb_ip/Bxb.v: $(CHISEL_FILES)
	cd ../../ && sbt "runMain bxb.Bxb --output-file $(abspath $@)"

soc_system.qsys: soc_system.tcl bxb_ip/Bxb.v
	qsys-script --search-path=bxb_ip/,$$ --script=soc_system.tcl

soc_system.done_stamp: soc_system.qsys bxb_ip/Bxb.v
	qsys-generate $< --search-path=bxb_ip/,$$ --synthesis=VERILOG
	touch soc_system.done_stamp
