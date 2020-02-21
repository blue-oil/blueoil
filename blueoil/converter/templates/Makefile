SRC_DIR := ./src
MAINS_DIR := ./mains
INPUTS_SRC_DIR := ./src/inputs
DLK_TEST_SRC_DIR := ./src/test_data

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

SRC := $(LIB_SRC) $(wildcard $(DLK_TEST_SRC_DIR)/*.cpp) mains/main.cpp
SRC := $(filter-out ./src/network_c_interface.cpp, $(SRC))

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
OBJ := $(patsubst %.cpp, %.o, $(SRC))

INCLUDES := -I./include


TARGETS_X86  := lm_x86

TARGETS_X86_AVX  := lm_x86_avx

TARGETS_AARCH64 := lm_aarch64

TARGETS_ARM  := lm_arm

TARGETS_FPGA := lm_fpga

LIBS_X86     := lib_x86

LIBS_X86_AVX := lib_x86_avx

LIBS_AARCH64 := lib_aarch64

LIBS_ARM     := lib_arm

LIBS_FPGA    := lib_fpga

ARS_X86     := ar_x86

ARS_X86_AVX := ar_x86_avx

ARS_AARCH64 := ar_aarch64

ARS_X86     := ar_x86

ARS_ARM     := ar_arm

ARS_FPGA    := ar_fpga


RM       := rm -rf

.PHONY: clean
clean:
	-$(RM) *.elf
	-$(RM) *.a
	-$(RM) *.so
	-$(RM) $(LIB_OBJ)
	-$(RM) $(LIB_X86_OBJ)
	-$(RM) $(LIB_X86_AVX_OBJ)
	-$(RM) $(LIB_ARM_OBJ)
	-$(RM) $(LIB_FPGA_OBJ)
	-$(RM) $(LIB_AARCH64_OBJ)
	-$(RM) $(OBJ)

lm_x86:           CXX = g++
lm_x86:           FLAGS += $(INCLUDES) -O3 -std=c++14 -DUSE_PNG -pthread -g
lm_x86:           CXXFLAGS +=

lm_x86_avx:       CXX = g++
lm_x86_avx:       FLAGS += $(INCLUDES) -O3 -std=c++14 -mavx2 -mfma -DUSE_AVX -DUSE_PNG -pthread -g -fopenmp
lm_x86_avx:       CXXFLAGS +=

lm_aarch64:       CXX = aarch64-linux-gnu-g++
lm_aarch64:       FLAGS += $(INCLUDES) -std=c++14 -O3 -DUSE_NEON -DUSE_PNG -pthread -g -fopenmp
lm_aarch64:       CXXFLAGS +=

lm_arm:           CXX = arm-linux-gnueabihf-g++
lm_arm:           FLAGS += $(INCLUDES) -std=c++14 -O3 -DUSE_NEON -DUSE_PNG -DAARCH32 -mcpu=cortex-a9 -mfpu=neon -mthumb -s -pthread -g -fopenmp
lm_arm:           CXXFLAGS +=

lm_fpga:          CXX = arm-linux-gnueabihf-g++
lm_fpga:          FLAGS += $(INCLUDES) -std=c++14 -O3 -DUSE_NEON -DRUN_ON_FPGA -DUSE_PNG -DAARCH32 -mcpu=cortex-a9 -mfpu=neon -mthumb -pthread -g -fopenmp -DFUNC_TIME_MEASUREMENT
lm_fpga:          CXXFLAGS +=

.PHONY: lib_x86 lib_x86_avx lib_aarch64 lib_arm lib_fpga

lib_x86:           CXX = g++
lib_x86:           FLAGS += $(INCLUDES) -O3 -std=c++14 -fPIC -fvisibility=hidden -pthread -g
lib_x86:           CXXFLAGS +=
lib_x86:           NAME = x86

lib_x86_avx:       CXX = g++
lib_x86_avx:       FLAGS += $(INCLUDES) -O3 -std=c++14 -fPIC -fvisibility=hidden -DUSE_AVX -mavx2 -mfma -pthread -g -fopenmp
lib_x86_avx:       CXXFLAGS +=
lib_x86_avx:       NAME = x86_avx

lib_aarch64:       CXX = aarch64-linux-gnu-g++
lib_aarch64:       FLAGS += $(INCLUDES) -O3 -std=c++14 -fPIC -fvisibility=hidden -DUSE_NEON -pthread -g
lib_aarch64:       CXXFLAGS +=
lib_aarch64:       NAME = aarch64

lib_arm:           CXX = arm-linux-gnueabihf-g++
lib_arm:           FLAGS += $(INCLUDES) -O3 -std=c++14 -fPIC -DUSE_NEON -DAARCH32 -mcpu=cortex-a9 -mfpu=neon -mthumb -fvisibility=hidden -pthread -g -fopenmp
lib_arm:           CXXFLAGS +=
lib_arm:           NAME = arm

lib_fpga:          CXX = arm-linux-gnueabihf-g++
lib_fpga:          FLAGS += $(INCLUDES) -O3 -std=c++14 -fPIC -DUSE_NEON -DRUN_ON_FPGA -DAARCH32 -mcpu=cortex-a9 -mfpu=neon -mthumb -fvisibility=hidden -pthread -g -fopenmp
lib_fpga:          CXXFLAGS +=
lib_fpga:          NAME = fpga

ar_x86:           AR = ar
ar_x86:           CXX = g++
ar_x86:           FLAGS += $(INCLUDES) -O3 -std=c++14 -fPIC -fvisibility=hidden -pthread -g
ar_x86:           LDFLAGS += -rcs
ar_x86:           NAME = x86

ar_x86_avx:       AR = ar
ar_x86_avx:       CXX = g++
ar_x86_avx:       FLAGS += $(INCLUDES) -O3 -std=c++14 -fPIC -fvisibility=hidden -DUSE_AVX -mavx2 -mfma -pthread -g -fopenmp
ar_x86_avx:       LDFLAGS += -rcs
ar_x86_avx:       NAME = x86_avx

ar_aarch64:       AR = aarch64-linux-gnu-ar
ar_aarch64:       CXX = aarch64-linux-gnu-g++
ar_aarch64:       FLAGS += $(INCLUDES) -O3 -std=c++14 -fPIC -fvisibility=hidden -DUSE_NEON -pthread -g
ar_aarch64:       LDFLAGS += -rcs
ar_aarch64:       NAME = aarch64

ar_arm:           AR = arm-linux-gnueabihf-ar
ar_arm:           CXX = arm-linux-gnueabihf-g++
ar_arm:           FLAGS += $(INCLUDES) -O3 -std=c++14 -fPIC -DUSE_NEON -DAARCH32 -mcpu=cortex-a9 -mfpu=neon -mthumb -fvisibility=hidden -pthread -g -fopenmp
ar_arm:           LDFLAGS += -rcs
ar_arm:           NAME = arm

ar_fpga:          AR = arm-linux-gnueabihf-ar
ar_fpga:          CXX = arm-linux-gnueabihf-g++
ar_fpga:          FLAGS += $(INCLUDES) -O3 -std=c++14 -fPIC -DUSE_NEON -DRUN_ON_FPGA -DAARCH32 -mcpu=cortex-a9 -mfpu=neon -mthumb -fvisibility=hidden -pthread -g -fopenmp
ar_fpga:          LDFLAGS += -rcs
ar_fpga:          NAME = fpga


$(TARGETS_ARM): $(OBJ) $(LIB_ARM_OBJ)
	$(CXX) $(FLAGS) $(OBJ) $(LIB_ARM_OBJ) -o $@.elf $(CXXFLAGS) -pthread -ldl

$(TARGETS_FPGA): $(OBJ) $(LIB_FPGA_OBJ)
	$(CXX) $(FLAGS) $(OBJ) $(LIB_FPGA_OBJ) -o $@.elf $(CXXFLAGS) -pthread -ldl

$(TARGETS_AARCH64): $(OBJ) $(LIB_AARCH64_OBJ)
	$(CXX) $(FLAGS) $(OBJ) $(LIB_AARCH64_OBJ) -o $@.elf $(CXXFLAGS) -pthread -ldl

$(TARGETS_X86): $(OBJ) $(LIB_X86_OBJ)
	$(CXX) $(FLAGS) $(OBJ) $(LIB_X86_OBJ) -o $@.elf $(CXXFLAGS) -pthread -ldl

$(TARGETS_X86_AVX): $(OBJ) $(LIB_X86_AVX_OBJ)
	$(CXX) $(FLAGS) $(OBJ) $(LIB_X86_AVX_OBJ) -o $@.elf $(CXXFLAGS) -pthread -ldl

$(LIBS_X86): $(LIB_OBJ) $(LIB_X86_OBJ)
	$(CXX) $(FLAGS) $(LIB_OBJ) $(LIB_X86_OBJ) -o libdlk_$(NAME).so $(CXXFLAGS) -shared -pthread -ldl

$(LIBS_X86_AVX): $(LIB_OBJ) $(LIB_X86_AVX_OBJ)
	$(CXX) $(FLAGS) $(LIB_OBJ) $(LIB_X86_AVX_OBJ) -o libdlk_$(NAME).so $(CXXFLAGS) -shared -pthread -ldl

$(LIBS_AARCH64): $(LIB_OBJ) $(LIB_AARCH64_OBJ)
	$(CXX) $(FLAGS) $(LIB_OBJ) $(LIB_AARCH64_OBJ) -o libdlk_$(NAME).so $(CXXFLAGS) -shared -pthread -ldl

$(LIBS_ARM): $(LIB_OBJ) $(LIB_ARM_OBJ)
	$(CXX) $(FLAGS) $(LIB_OBJ) $(LIB_ARM_OBJ) -o libdlk_$(NAME).so $(CXXFLAGS) -shared -pthread -ldl

$(LIBS_FPGA): $(LIB_OBJ) $(LIB_FPGA_OBJ)
	$(CXX) $(FLAGS) $(LIB_OBJ) $(LIB_FPGA_OBJ) -o libdlk_$(NAME).so $(CXXFLAGS) -shared -pthread -ldl

$(ARS_X86): $(LIB_OBJ) $(LIB_X86_OBJ)
	$(AR) $(LDFLAGS) libdlk_$(NAME).a $(LIB_OBJ) $(LIB_X86_OBJ)

$(ARS_X86_AVX): $(LIB_OBJ) $(LIB_X86_AVX_OBJ)
	$(AR) $(LDFLAGS) libdlk_$(NAME).a $(LIB_OBJ) $(LIB_X86_AVX_OBJ)

$(ARS_AARCH64): $(LIB_OBJ) $(LIB_AARCH64_OBJ)
	$(AR) $(LDFLAGS) libdlk_$(NAME).a $(LIB_OBJ) $(LIB_AARCH64_OBJ)

$(ARS_ARM): $(LIB_OBJ) $(LIB_ARM_OBJ)
	$(AR) $(LDFLAGS) libdlk_$(NAME).a $(LIB_OBJ) $(LIB_ARM_OBJ)

$(ARS_FPGA): $(LIB_OBJ) $(LIB_FPGA_OBJ)
	$(AR) $(LDFLAGS) libdlk_$(NAME).a $(LIB_OBJ) $(LIB_FPGA_OBJ)

%.o: %.S
	$(CXX) $(FLAGS) -c $^ -o $@ $(CXXFLAGS)

%.o: %.cpp
	$(CXX) $(FLAGS) -c $^ -o $@ $(CXXFLAGS)
