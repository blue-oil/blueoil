CHISEL_FILES := $(wildcard ../../src/main/scala/**/*.scala)


.NOTPARALLEL: all

all:  rtl/Bxb.v vivado vivado_gen

rtl/Bxb.v: $(CHISEL_FILES)
	cd ../../ && sbt "runMain bxb.Bxb --output-file $(abspath $@)"

.PHONY: vivado
vivado: 
	vivado -mode tcl -source bxb.tcl
#	rm -rf project_1/
#	rm -rf vivado*

.PHONY: vivado_gen
vivado_gen:
	vivado -mode tcl -source ipv2.tcl
	bootgen -image bin/ipv2.bif -arch zynqmp -w -o bin/ipv2.bin

.PHONY: clean
clean:
	rm -rf rtl/Bxb.v
	rm -rf myproj/
	rm -rf project_1/
	rm -rf vivado*
	rm -rf rtl/component.xml rtl/xgui
	rm -rf bin/*.bin bin/*.bit

