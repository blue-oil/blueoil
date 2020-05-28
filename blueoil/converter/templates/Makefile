SRC_DIR := ./src
MAINS_DIR := ./mains
INPUTS_SRC_DIR := ./src/inputs

TYPE_ARGUMENT := {executable, dynamic, static}

.PHONY: help
help:
	@echo make clean
	@echo make build ARCH=x86 TYPE=$(TYPE_ARGUMENT) 
	@echo make build ARCH=x86_avx TYPE=$(TYPE_ARGUMENT) 
	@echo make build ARCH=aarch64 TYPE=$(TYPE_ARGUMENT) 
	@echo make build ARCH=aarch64 USE_FPGA TYPE=$(TYPE_ARGUMENT) 
	@echo make build ARCH=arm TYPE=$(TYPE_ARGUMENT) 
	@echo make build ARCH=arm USE_FPGA TYPE=$(TYPE_ARGUMENT) 

LIB_SRC := $(wildcard $(INPUTS_SRC_DIR)/*.cpp) \
    $(wildcard $(SRC_DIR)/scaling_factors.cpp) \
    $(wildcard $(SRC_DIR)/thresholds.cpp) \
    $(SRC_DIR)/func/average_pool.cpp \
    $(SRC_DIR)/func/conv2d.cpp \
    $(SRC_DIR)/func/max_pool.cpp \
    $(SRC_DIR)/func/pad.cpp \
    $(SRC_DIR)/func/matmul.cpp \
    $(SRC_DIR)/func/softmax.cpp \
    $(SRC_DIR)/func/lookup.cpp \
    $(SRC_DIR)/matrix/shift_add.cpp \
    $(SRC_DIR)/matrix/multiplication.cpp \
    $(SRC_DIR)/network_c_interface.cpp \
    $(SRC_DIR)/network.cpp \
    $(SRC_DIR)/pack_input_to_qwords.cpp \
    $(SRC_DIR)/time_measurement.cpp \
    $(SRC_DIR)/write_to_file.cpp \
    $(SRC_DIR)/quantizer.cpp \
    $(SRC_DIR)/tensor_save.cpp

MAIN_SRC := $(LIB_SRC) mains/main.cpp
MAIN_SRC := $(filter-out ./src/network_c_interface.cpp, $(MAIN_SRC))

LIB_ARM_SRC := $(wildcard $(SRC_DIR)/*.S) \
    $(SRC_DIR)/func/arm_neon/batch_normalization.cpp \
    $(SRC_DIR)/func/impl/arm_neon/quantized_conv2d_tiling.cpp \
    $(SRC_DIR)/func/impl/arm_neon/pop_count.cpp
LIB_ARM_OBJ := $(patsubst %.S, %.o, $(LIB_ARM_SRC))
LIB_ARM_OBJ := $(patsubst %.cpp, %.o, $(LIB_ARM_OBJ))

LIB_FPGA_SRC := $(wildcard $(SRC_DIR)/*.S) \
    $(SRC_DIR)/func/arm_neon/batch_normalization.cpp \
    $(SRC_DIR)/func/impl/fpga/quantized_conv2d_accelerator.cpp \
    $(SRC_DIR)/func/impl/arm_neon/pop_count.cpp
LIB_FPGA_OBJ := $(patsubst %.S, %.o, $(LIB_FPGA_SRC))
LIB_FPGA_OBJ := $(patsubst %.cpp, %.o, $(LIB_FPGA_OBJ))

LIB_AARCH64_SRC := \
    $(SRC_DIR)/func/arm_neon/batch_normalization.cpp \
    $(SRC_DIR)/func/impl/arm_neon/quantized_conv2d_tiling.cpp \
    $(SRC_DIR)/func/impl/arm_neon/pop_count.cpp
LIB_AARCH64_OBJ := $(patsubst %.S, %.o, $(LIB_AARCH64_SRC))
LIB_AARCH64_OBJ := $(patsubst %.cpp, %.o, $(LIB_AARCH64_OBJ))

LIB_AARCH64_FPGA_SRC := \
    $(SRC_DIR)/func/arm_neon/batch_normalization.cpp \
    $(SRC_DIR)/func/impl/fpga/quantized_conv2d_accelerator.cpp \
    $(SRC_DIR)/func/impl/arm_neon/pop_count.cpp
LIB_AARCH64_FPGA_OBJ := $(patsubst %.S, %.o, $(LIB_AARCH64_FPGA_SRC))
LIB_AARCH64_FPGA_OBJ := $(patsubst %.cpp, %.o, $(LIB_AARCH64_FPGA_OBJ))

LIB_X86_SRC := \
    $(SRC_DIR)/func/generic/batch_normalization.cpp \
    $(SRC_DIR)/func/impl/generic/quantized_conv2d_kn2row.cpp \
    $(SRC_DIR)/matrix/generic/quantized_multiplication.cpp \
    $(SRC_DIR)/func/impl/generic/pop_count.cpp \
    $(SRC_DIR)/func/impl/generic/apply_thresholds.cpp \
    $(SRC_DIR)/func/impl/generic/pack_16bit.cpp
LIB_X86_OBJ := $(patsubst %.cpp, %.o, $(LIB_X86_SRC))

LIB_X86_AVX_SRC := \
    $(SRC_DIR)/func/x86_avx/batch_normalization.cpp \
    $(SRC_DIR)/func/impl/x86_avx/quantized_conv2d_tiling.cpp \
    $(SRC_DIR)/func/impl/generic/pop_count.cpp
LIB_X86_AVX_OBJ := $(patsubst %.cpp, %.o, $(LIB_X86_AVX_SRC))

LIB_OBJ := $(patsubst %.cpp, %.o, $(LIB_SRC))
MAIN_OBJ := $(patsubst %.cpp, %.o, $(MAIN_SRC))

INCLUDES := -I./include

ALL_OBJ := \
    $(MAIN_OBJ) $(LIB_OBJ) \
    $(LIB_X86_OBJ) $(LIB_X86_AVX_OBJ) \
    $(LIB_AARCH64_OBJ) $(LIB_AARCH64_FPGA_OBJ) \
    $(LIB_ARM_OBJ) $(LIB_FPGA_OBJ)

ALL_DEP := $(ALL_OBJ:.o=.d)
-include $(ALL_DEP)

RM       := rm -rf

.PHONY: clean
clean:
	-$(RM) *.elf
	-$(RM) *.a
	-$(RM) *.so
	-$(RM) $(ALL_OBJ)
	-$(RM) $(ALL_DEP)


# Base Settings
CXXFLAGS := -I./include -O3 -std=c++14 -g -MMD -MP
ARFLAGS := -rcs
LDFLAGS += -pthread -ldl
BASE_NAME = $(ARCH)

################################
# Set variables by ARCH
################################
ifeq ($(ARCH), x86)
CXX := g++
CXXFLAGS +=
OBJ := $(LIB_X86_OBJ)
AR = ar
endif

ifeq ($(ARCH), x86_avx)
CXX := g++
CXXFLAGS += -mavx2 -mfma -DUSE_AVX -fopenmp
OBJ := $(LIB_X86_AVX_OBJ)
AR = ar
endif

ifeq ($(ARCH), aarch64)
CXX := aarch64-linux-gnu-g++
CXXFLAGS += -DUSE_NEON -fopenmp
AR = aarch64-linux-gnu-ar

ifndef USE_FPGA
OBJ := $(LIB_AARCH64_OBJ)
else
OBJ := $(LIB_AARCH64_FPGA_OBJ)
CXXFLAGS += -DRUN_ON_FPGA
BASE_NAME += _fpga
endif

endif

ifeq ($(ARCH), arm)
CXX := arm-linux-gnueabihf-g++
CXXFLAGS += -DUSE_NEON -DAARCH32 -mcpu=cortex-a9 -mfpu=neon -mthumb -fopenmp
AR = arm-linux-gnueabihf-ar

ifndef USE_FPGA
OBJ := $(LIB_ARM_OBJ)
else
OBJ := $(LIB_FPGA_OBJ)
CXXFLAGS += -DRUN_ON_FPGA
BASE_NAME += _fpga
endif

endif

################################
# Set variables by TYPE
################################
ifeq ($(TYPE), executable)
OBJ += $(MAIN_OBJ)
CXXFLAGS += -DUSE_PNG -DFUNC_TIME_MEASUREMENT
LDFLAGS +=
NAME := lm_$(BASE_NAME).elf
LINK := $(CXX) $(OBJ) -o $(NAME) $(CXXFLAGS) $(LDFLAGS)
endif

ifeq ($(TYPE), dynamic)
OBJ += $(LIB_OBJ)
CXXFLAGS += -fPIC -fvisibility=hidden
LDFLAGS += -shared
NAME := libdlk_$(BASE_NAME).so
LINK := $(CXX) $(OBJ) -o $(NAME) $(CXXFLAGS) $(LDFLAGS)
endif

ifeq ($(TYPE), static)
OBJ += $(LIB_OBJ)
CXXFLAGS += -fPIC -fvisibility=hidden
LDFLAGS +=
NAME := libdlk_$(BASE_NAME).a
LINK := $(AR) $(ARFLAGS) $(NAME) $(OBJ)
endif


.PHONY: build
build: $(NAME)
	@echo Building $(NAME)

$(NAME): $(OBJ)
	$(LINK)

%.o: %.S
	$(CXX) -c $< -o $@ $(CXXFLAGS)

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

