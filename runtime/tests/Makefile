CXX = g++ -std=c++11
INCLUDES = -I../yaml-cpp/include -I../include/ `pkg-config --cflags opencv`
LIBS = ../yaml-cpp/build/libyaml-cpp.a `pkg-config --cflags --libs opencv`
CXXFLAGS = -g -W -Wall -Wextra $(INCLUDES)
LDFLAGS = $(LIBS)
PROGS = test_tensor test_resize
MAIN_SRC = test_tensor.cpp test_resize.cpp
MAIN_OBJS = $(MAIN_SRCS:.cpp=.o)
SRCS = network_dummy.cpp ../src/blueoil.cpp ../src/blueoil_data_processor.cpp
OBJS := $(SRCS:.cpp=.o)

all: $(PROGS)
	for t in $(PROGS) ; do echo running $$t ; ./$$t ; done

$(PROGS): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $@.cpp $(OBJS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(PROGS) $(MAIN_OBJS)
