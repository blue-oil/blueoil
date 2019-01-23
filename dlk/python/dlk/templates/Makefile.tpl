DEVICE := CycloneV
CLOCK  := {{ config.ip_frequency }}

COMPONENTS := --component binary_convolution_hls

SRC_DIR := ./src
MAINS_DIR := ./mains
INPUTS_SRC_DIR := ./src/inputs
DLK_TEST_SRC_DIR := ./src/test_data

LIB_SRC := $(wildcard $(INPUTS_SRC_DIR)/*.cpp) \
{%- if config.activate_hard_quantization %}
    $(SRC_DIR)/scaling_factors.cpp \
{%- endif %}
{%- if config.threshold_skipping %}
    $(SRC_DIR)/thresholds.cpp \
{%- endif %}
    $(SRC_DIR)/func/add.cpp \
    $(SRC_DIR)/func/average_pool.cpp \
    $(SRC_DIR)/func/bias_add.cpp \
    $(SRC_DIR)/func/conv2d.cpp \
    $(SRC_DIR)/func/apply_thresholds.cpp \
    $(SRC_DIR)/func/max.cpp \
    $(SRC_DIR)/func/max_pool.cpp \
    $(SRC_DIR)/func/minimum.cpp \
    $(SRC_DIR)/func/mul.cpp \
    $(SRC_DIR)/func/quantize.cpp \
    $(SRC_DIR)/func/real_div.cpp \
    $(SRC_DIR)/func/relu.cpp \
    $(SRC_DIR)/func/round.cpp \
    $(SRC_DIR)/func/scale.cpp \
    $(SRC_DIR)/func/softmax.cpp \
    $(SRC_DIR)/func/sqrt.cpp \
    $(SRC_DIR)/func/sub.cpp \
    $(SRC_DIR)/func/unpooling.cpp \
    $(SRC_DIR)/network_c_interface.cpp \
    $(SRC_DIR)/network.cpp \
    $(SRC_DIR)/pack_input_to_qwords.cpp \
    $(SRC_DIR)/time_measurement.cpp

SRC := $(LIB_SRC) $(wildcard $(DLK_TEST_SRC_DIR)/*.cpp) mains/main.cpp
SRC := $(filter-out ./src/network_c_interface.cpp, $(SRC))

LIB_ARM_SRC := $(wildcard $(SRC_DIR)/asm/arch/arm_neon/*.S) \
    $(SRC_DIR)/func/arch/generic/quantized_conv2d_dim2col.cpp \
    $(SRC_DIR)/func/arch/arm_neon/quantized_conv2d.cpp \
    $(SRC_DIR)/func/arch/arm_neon/batch_normalization.cpp \
    $(SRC_DIR)/func/arch/arm_neon/quantized_conv2d_tiling.cpp \
    $(SRC_DIR)/func/arch/generic/quantized_conv2d_kn2row.cpp \
    $(SRC_DIR)/matrix/arm_neon/quantized_multiplication.cpp \
    $(SRC_DIR)/matrix/arm_neon/shift_add.cpp
LIB_ARM_OBJ := $(patsubst %.S, %.o, $(LIB_ARM_SRC))
LIB_ARM_OBJ := $(patsubst %.cpp, %.o, $(LIB_ARM_OBJ))

LIB_FPGA_SRC := $(wildcard $(SRC_DIR)/asm/arch/arm_neon/*.S) \
    $(SRC_DIR)/func/arch/generic/quantized_conv2d_dim2col.cpp \
    $(SRC_DIR)/func/arch/fpga/quantized_conv2d.cpp \
    $(SRC_DIR)/func/arch/arm_neon/batch_normalization.cpp \
    $(SRC_DIR)/func/arch/fpga/quantized_conv2d_kn2row.cpp \
    $(SRC_DIR)/matrix/arm_neon/quantized_multiplication.cpp \
    $(SRC_DIR)/matrix/arm_neon/shift_add.cpp    
LIB_FPGA_OBJ := $(patsubst %.S, %.o, $(LIB_FPGA_SRC))
LIB_FPGA_OBJ := $(patsubst %.cpp, %.o, $(LIB_FPGA_OBJ))

LIB_AARCH64_SRC := $(wildcard $(SRC_DIR)/asm/arch/aarch64_neon/*.S) \
    $(SRC_DIR)/func/arch/generic/quantized_conv2d_dim2col.cpp \
    $(SRC_DIR)/func/arch/aarch64_neon/quantized_conv2d.cpp \
    $(SRC_DIR)/func/arch/aarch64_neon/batch_normalization.cpp \
    $(SRC_DIR)/func/arch/aarch64_neon/quantized_conv2d_tiling.cpp \
    $(SRC_DIR)/func/arch/generic/quantized_conv2d_kn2row.cpp \
    $(SRC_DIR)/matrix/aarch64_neon/quantized_multiplication.cpp \
    $(SRC_DIR)/matrix/aarch64_neon/shift_add.cpp    
LIB_AARCH64_OBJ := $(patsubst %.S, %.o, $(LIB_AARCH64_SRC))
LIB_AARCH64_OBJ := $(patsubst %.cpp, %.o, $(LIB_AARCH64_OBJ))

LIB_X86_SRC := \
    $(SRC_DIR)/func/arch/generic/quantized_conv2d_dim2col.cpp \
    $(SRC_DIR)/func/arch/generic/quantized_conv2d.cpp \
    $(SRC_DIR)/func/arch/generic/batch_normalization.cpp \
    $(SRC_DIR)/func/arch/generic/quantized_conv2d_kn2row.cpp \
    $(SRC_DIR)/matrix/generic/quantized_multiplication.cpp \
    $(SRC_DIR)/matrix/generic/shift_add.cpp    
LIB_X86_OBJ := $(patsubst %.cpp, %.o, $(LIB_X86_SRC))

LIB_OBJ := $(patsubst %.cpp, %.o, $(LIB_SRC))
OBJ := $(patsubst %.cpp, %.o, $(SRC))

INCLUDES := -I./include
HLS_INCLUDE := -I./hls/include


TARGETS_X86  := lm_x86

TARGETS_AARCH64 := lm_aarch64

TARGETS_ARM  := lm_arm

TARGETS_FPGA := lm_fpga

LIBS_X86     := lib_x86

LIBS_AARCH64 := lib_aarch64

LIBS_ARM     := lib_arm

LIBS_FPGA    := lib_fpga

ARS_X86     := ar_x86

ARS_AARCH64 := ar_aarch64

ARS_X86     := ar_x86

ARS_ARM     := ar_arm

ARS_FPGA    := ar_fpga
		   

RM_TARGETS_LIST := $(TARGETS) \
                   hls_simulation \
                   hls_synthesis \
                   derive_threshold
RM       := rm -rf

HLS_INSTALL_DIR := $(shell which i++ | sed 's|/bin/i++||g')

.PHONY: test
test: $(TARGETS)
	@$(foreach t,$(TARGETS),echo ./$(t); ./$(t); echo "";)

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	-$(RM) $(TARGETS) $(foreach t,$(RM_TARGETS_LIST),$(t).elf) transcript
	-$(RM) *.a
	-$(RM) $(LIB_OBJ)
	-$(RM) $(LIB_X86_OBJ)
	-$(RM) $(LIB_ARM_OBJ)
	-$(RM) $(LIB_FPGA_OBJ)
	-$(RM) $(LIB_AARCH64_OBJ)
	-$(RM) $(OBJ)

lm_x86:           CXX = /usr/local/clang+llvm-7.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/clang++
lm_x86:           FLAGS += $(INCLUDES) -std=c++0x -O3 -DUSE_PNG -pthread -g -fopenmp=libgomp -march=native -I/usr/local/clang+llvm-7.0.0-x86_64-linux-gnu-ubuntu-16.04/include/c++/v1/ -I/usr/local/clang+llvm-7.0.0-x86_64-linux-gnu-ubuntu-16.04/lib/clang/7.0.0/include/ -L/usr/local/clang+llvm-7.0.0-x86_64-linux-gnu-ubuntu-16.04/lib/ -fuse-ld=/usr/local/clang+llvm-7.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/ld.lld -flto=thin -Wl,--lto-O3 -stdlib=libc++ -lc++abi -static -Wno-everything -Wl,-s

lm_aarch64:           CXX = /usr/local/clang+llvm-7.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/clang++
lm_aarch64:           FLAGS += $(INCLUDES) -std=c++0x -O3 -DUSE_ASIMD -DUSE_PNG -pthread -g -fopenmp=libgomp --target=aarch64-linux-gnu -I/usr/local/clang+llvm-7.0.0-aarch64-linux-gnu/include/c++/v1/ -I/usr/aarch64-linux-gnu/include/ -I/usr/local/clang+llvm-7.0.0-aarch64-linux-gnu/lib/clang/7.0.0/include/ -L/usr/local/clang+llvm-7.0.0-aarch64-linux-gnu/lib/ -fuse-ld=/usr/local/clang+llvm-7.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/ld.lld -flto=thin -Wl,--lto-O3 -stdlib=libc++ -lc++abi -static -Wno-everything -Wl,-s

lm_arm:           CXX = /usr/local/clang+llvm-7.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/clang++
lm_arm:           FLAGS += $(INCLUDES) -std=c++0x -O3 -DUSE_NEON -DUSE_PNG -D_OPENMP -mfpu=neon -mcpu=cortex-a9 -mthumb -pthread -g -fopenmp=libgomp --target=arm-linux-gnueabihf -mfloat-abi=hard -I/usr/local/clang+llvm-7.0.0-armv7a-linux-gnueabihf/include/c++/v1/ -I/usr/arm-linux-gnueabihf/include/ -I/usr/local/clang+llvm-7.0.0-armv7a-linux-gnueabihf/lib/clang/7.0.0/include/ -L/usr/local/clang+llvm-7.0.0-armv7a-linux-gnueabihf/lib/ -fuse-ld=/usr/local/clang+llvm-7.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/ld.lld -flto=thin -Wl,--lto-O3 -stdlib=libc++ -lc++abi -static -Wno-everything -Wl,-s
lm_arm:           CXXFLAGS +=

lm_fpga:          CXX = /usr/local/clang+llvm-7.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/clang++
lm_fpga:          FLAGS += $(INCLUDES) -std=c++0x -O3 -DUSE_NEON -DRUN_ON_FPGA -DUSE_PNG -D_OPENMP -mfpu=neon -mcpu=cortex-a9 -mthumb -pthread -g -fopenmp=libgomp --target=arm-linux-gnueabihf -mfloat-abi=hard -I/usr/local/clang+llvm-7.0.0-armv7a-linux-gnueabihf/include/c++/v1/ -I/usr/arm-linux-gnueabihf/include/ -I/usr/local/clang+llvm-7.0.0-armv7a-linux-gnueabihf/lib/clang/7.0.0/include/ -L/usr/local/clang+llvm-7.0.0-armv7a-linux-gnueabihf/lib/ -fuse-ld=/usr/local/clang+llvm-7.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/ld.lld -flto=thin -Wl,--lto-O3 -stdlib=libc++ -lc++abi -static -Wno-everything -Wl,-s
lm_fpga:          CXXFLAGS +=

lib_x86:           CXX = g++
lib_x86:           FLAGS += $(INCLUDES) -O3 -std=c++0x -fPIC -fvisibility=hidden -pthread -g -Wl,-s

lib_aarch64:       CXX = aarch64-linux-gnu-g++
lib_aarch64:       FLAGS += $(INCLUDES) -O3 -std=c++0x -fPIC -DUSE_NEON -fvisibility=hidden -pthread -g -Wl,-s

lib_arm:           CXX = arm-linux-gnueabihf-g++
lib_arm:           FLAGS += $(INCLUDES) -O3 -std=c++0x -fPIC -DUSE_NEON -mcpu=cortex-a9 -mfpu=neon -mthumb -fvisibility=hidden -pthread -g -fopenmp -Wl,-s
lib_arm:           CXXFLAGS +=

lib_fpga:          CXX = arm-linux-gnueabihf-g++
lib_fpga:          FLAGS += $(INCLUDES) -O3 -std=c++0x -fPIC -DUSE_NEON -DRUN_ON_FPGA -mcpu=cortex-a9 -mfpu=neon -mthumb -fvisibility=hidden -pthread -g -fopenmp -Wl,-s
lib_fpga:          CXXFLAGS +=

ar_x86:           AR = ar
ar_x86:           CXX = g++
ar_x86:           FLAGS += $(INCLUDES) -O3 -std=c++0x -fPIC -fvisibility=hidden -pthread -g
ar_x86:           LDFLAGS += -rcs
ar_x86:           NAME = x86

ar_aarch64:           AR = aarch64-linux-gnu-ar
ar_aarch64:           CXX = aarch64-linux-gnu-g++
ar_aarch64:           FLAGS += $(INCLUDES) -O3 -std=c++0x -fPIC -DUSE_NEON -fvisibility=hidden -pthread -g
ar_aarch64:           LDFLAGS += -rcs
ar_aarch64:           NAME = aarch64

ar_arm:           AR = arm-linux-gnueabihf-ar
ar_arm:           CXX = arm-linux-gnueabihf-g++
ar_arm:           FLAGS += $(INCLUDES) -O3 -std=c++0x -fPIC -DUSE_NEON -mcpu=cortex-a9 -mfpu=neon -mthumb -fvisibility=hidden -pthread -g -fopenmp
ar_arm:           LDFLAGS += -rcs
ar_arm:           NAME = arm

ar_fpga:          AR = arm-linux-gnueabihf-ar
ar_fpga:          CXX = arm-linux-gnueabihf-g++
ar_fpga:          FLAGS += $(INCLUDES) -O3 -std=c++0x -fPIC -DUSE_NEON -DRUN_ON_FPGA -mcpu=cortex-a9 -mfpu=neon -mthumb -fvisibility=hidden -pthread -g -fopenmp
ar_fpga:          LDFLAGS += -rcs
ar_fpga:          NAME = fpga


$(TARGETS_ARM): $(OBJ) $(TVM_OBJ) $(LIB_ARM_OBJ)
	$(CXX) $(FLAGS) $(OBJ) $(TVM_OBJ) $(LIB_ARM_OBJ) -o $@.elf $(CXXFLAGS) $(TVM_ARM_LIBS) -pthread -ldl

$(TARGETS_FPGA): $(OBJ) $(TVM_OBJ) $(LIB_FPGA_OBJ)
	$(CXX) $(FLAGS) $(OBJ) $(TVM_OBJ) $(LIB_FPGA_OBJ) -o $@.elf $(CXXFLAGS) $(TVM_ARM_LIBS) -pthread -ldl

$(TARGETS_AARCH64): $(OBJ) $(TVM_OBJ) $(LIB_AARCH64_OBJ)
	$(CXX) $(FLAGS) $(OBJ) $(TVM_OBJ) $(LIB_AARCH64_OBJ) -o $@.elf $(CXXFLAGS) $(TVM_AARCH64_OBJ) -pthread -ldl

$(TARGETS_X86): $(OBJ) $(TVM_OBJ) $(LIB_X86_OBJ)
	$(CXX) $(FLAGS) $(OBJ) $(TVM_OBJ) $(LIB_X86_OBJ) -o $@.elf $(CXXFLAGS) $(TVM_X86_LIBS) -pthread -ldl

$(LIBS_X86): $(LIB_OBJ) $(TVM_OBJ) $(LIB_X86_OBJ)
	$(CXX) $(FLAGS) $(LIB_OBJ) $(TVM_OBJ) $(LIB_X86_OBJ) -o $@.so $(CXXFLAGS) $(TVM_X86_LIBS) -shared -pthread -ldl

$(LIBS_AARCH64): $(LIB_OBJ) $(TVM_OBJ) $(LIB_AARCH64_OBJ)
	$(CXX) $(FLAGS) $(LIB_OBJ) $(TVM_OBJ) $(LIB_AARCH64_OBJ) -o $@.so $(CXXFLAGS) $(TVM_AARCH64_LIBS)  -shared -pthread -ldl

$(LIBS_ARM): $(LIB_OBJ) $(TVM_OBJ) $(LIB_ARM_OBJ)
	$(CXX) $(FLAGS) $(LIB_OBJ) $(TVM_OBJ) $(LIB_ARM_OBJ) -o $@.so $(CXXFLAGS) $(TVM_ARM_LIBS) -shared -pthread -ldl

$(LIBS_FPGA): $(LIB_OBJ) $(TVM_OBJ) $(LIB_FPGA_OBJ)
	$(CXX) $(FLAGS) $(LIB_OBJ) $(TVM_OBJ) $(LIB_FPGA_OBJ) -o $@.so $(CXXFLAGS) $(TVM_ARM_LIBS) -shared -pthread -ldl

$(ARS_X86): $(LIB_OBJ) $(TVM_OBJ) $(LIB_X86_OBJ)
	$(AR) $(LDFLAGS) libdlk_$(NAME).a $(LIB_OBJ) $(TVM_OBJ) $(TVM_X86_LIBS) $(LIB_X86_OBJ)

$(ARS_AARCH64): $(LIB_OBJ) $(TVM_OBJ) $(LIB_AARCH64_OBJ)
	$(AR) $(LDFLAGS) libdlk_$(NAME).a $(LIB_OBJ) $(TVM_OBJ) $(TVM_AARCH64_LIBS) $(LIB_AARCH64_OBJ)

$(ARS_ARM): $(LIB_OBJ) $(TVM_OBJ) $(LIB_ARM_OBJ)
	$(AR) $(LDFLAGS) libdlk_$(NAME).a $(LIB_OBJ) $(TVM_OBJ) $(LIB_ARM_OBJ) $(TVM_ARM_LIBS)

$(ARS_FPGA): $(LIB_OBJ) $(TVM_OBJ) $(LIB_FPGA_OBJ)
	$(AR) $(LDFLAGS) libdlk_$(NAME).a $(LIB_OBJ) $(TVM_OBJ) $(LIB_FPGA_OBJ) $(TVM_ARM_LIBS)

%.o: %.S
	$(CXX) $(FLAGS) -c $^ -o $@ $(CXXFLAGS)

%.o: %.cpp
	$(CXX) $(FLAGS) -c $^ -o $@ $(CXXFLAGS)
