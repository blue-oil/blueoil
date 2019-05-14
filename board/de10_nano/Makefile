CHISEL_FILES := $(wildcard ../../src/main/scala/**/*.scala)

.PHONY: bxb_ip
bxb_ip: bxb_ip/Bxb.v

.PHONY: soc_system
soc_system: soc_system.done_stamp

.PHONY: quartus
quartus: quartus.done_stamp

.PHONY: bitstream
bitstream: output_files/soc_system.rbf

.PHONY: preloader
preloader: software/spl_bsp/preloader-mkpimage.bin

bxb_ip/Bxb.v: $(CHISEL_FILES)
	cd ../../ && sbt "runMain bxb.Bxb --output-file $(abspath $@)"

soc_system.qsys: soc_system.tcl bxb_ip/Bxb.v
	qsys-script --search-path=bxb_ip/,$$ --script=soc_system.tcl

soc_system.done_stamp: soc_system.qsys bxb_ip/Bxb.v
	qsys-generate $< --search-path=bxb_ip/,$$ --synthesis=VERILOG
	touch soc_system.done_stamp

output_files/DE10_NANO_SoC_GHRD.sof quartus.done_stamp: DE10_NANO_SoC_GHRD.qpf DE10_NANO_SoC_GHRD.qsf DE10_NANO_SoC_GHRD.v DE10_NANO_SOC_GHRD.sdc soc_system.done_stamp 
	quartus_sh --flow compile DE10_NANO_SoC_GHRD.qpf
	touch quartus.done_stamp

output_files/soc_system.rbf: output_files/DE10_NANO_SoC_GHRD.sof
	quartus_cpf -c -o bitstream_compression=on $< $@

software/spl_bsp/preloader-mkpimage.bin: quartus.done_stamp
	bsp-create-settings --type spl --bsp-dir $(dir $@) --settings settings.bsp --preloader-settings-dir hps_isw_handoff/soc_system_hps_0
	$(MAKE) -j -C $(dir $@)
