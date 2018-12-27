DEVICE := 5CSEBA6U23I7
CLOCK  := 110MHz

CXX :=
COMPONENTS := --component intel_hls_qconv_kn2row_tiling_impl

SRC_DIR := ./src
TB_DIR := $(SRC_DIR)/tb
CPP_DIR := $(SRC_DIR)/cpp

TB_SRC := $(wildcard $(TB_DIR)/*.cpp)
TB_OBJ := $(patsubst %.cpp, %.o, $(TB_SRC))

CPP_SRC := $(wildcard $(CPP_DIR)/*.cpp)  
CPP_OBJ := $(patsubst %.cpp, %.o, $(CPP_SRC))
CPP_OBJ += $(TB_OBJ)


INCLUDE := -I./include

TARGETS  := x86 \
            arm \
            fpga

RM_TARGETS_LIST := $(TARGETS)
RM       := rm -rf


INTEL_HLS_DIR := $(SRC_DIR)/intel_hls

INTEL_HLS_INSTALL_DIR := $(shell which i++ | sed 's|/bin/i++||g')
INTEL_HLS_SRC := $(wildcard $(INTEL_HLS_DIR)/*.cpp)  
INTEL_HLS_OBJ := $(patsubst %.cpp, %.o, $(INTEL_HLS_SRC)) $(CPP_OBJ)
INTEL_HLS_INCLUDE := $(INCLUDE)


.PHONY: test
test: $(TARGETS)
	@$(foreach t,$(TARGETS),echo ./$(t); ./$(t); echo "";)

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	-$(RM) $(foreach t, $(RM_TARGETS_LIST), $(t).elf)
	-$(RM) $(TB_DIR)/*.o $(CPP_DIR)/*.o $(INTEL_HLS_DIR)/*.o


x86:	CXX = g++
x86:	FLAGS += $(INCLUDE) -O0 -std=c++0x -g

arm:	CXX = arm-linux-gnueabihf-g++
arm:    FLAGS += $(INCLUDE) -static -s -std=c++0x -O3

fpga:	CXX = arm-linux-gnueabihf-g++
fpga:	FLAGS += $(INCLUDE) -static -std=c++0x -s -O3 -D_DE10_NANO_

sim:   CXX = i++
sim:   FLAGS += -march=x86-64 $(INTEL_HLS_INCLUDE) -lm $(COMPONENTS) -D_INTEL_HLS_
sim:   $(INTEL_HLS_OBJ)
	$(CXX) $(FLAGS) $(INTEL_HLS_OBJ) -o $@.elf $(CXXFLAGS)

syn:    CXX = i++
syn:    FLAGS += -march=$(DEVICE) -I$(INTEL_HLS_INSTALL_DIR)/include -L$(INTEL_HLS_INSTALL_DIR)/host/linux64/lib $(INTEL_HLS_INCLUDE) $(COMPONENTS) -ghdl --clock $(CLOCK) -D_INTEL_HLS_
syn:   $(INTEL_HLS_OBJ)
	$(CXX) $(FLAGS) $(INTEL_HLS_OBJ) -o $@.elf $(CXXFLAGS)

$(TARGETS): $(CPP_OBJ)
	$(CXX) $(FLAGS) $(CPP_OBJ) -o $@.elf $(CXXFLAGS)

%.o: %.cpp
	$(CXX) $(FLAGS) -c $^ -o $@ $(CXXFLAGS)
