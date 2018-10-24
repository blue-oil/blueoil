CXX = g++ -std=c++11
INCLUDES = -I../include/ -I../yaml-cpp/include
LIBS = `pkg-config --cflags --libs opencv` `pkg-config --cflags --libs yaml-cpp`
CXXFLAGS = -g -W -Wall -Wextra $(INCLUDES)
LDFLAGS = $(LIBS)
PROGS = test_tensor
MAIN_SRC = test_tensor.cpp
MAIN_OBJS = $(MAIN_SRCS:.cpp=.o)
SRCS = network_dummy.cpp ../src/blueoil.cpp ../src/blueoil_data_processor.cpp
OBJS := $(SRCS:.cpp=.o)

all: $(PROGS)
	for t in $(PROGS) ; do echo running $$t ; ./$$t ; done

$(PROGS): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $@.cpp $(OBJS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(PROGS)
