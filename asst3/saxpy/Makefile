
EXECUTABLE := cudaSaxpy

CU_FILES   := saxpy.cu

CU_DEPS    :=

CC_FILES   := main.cpp
DEBUG = -DDEBUG			#-D 是编译器的一个标准选项，用于定义预处理宏,等同于在代码中添加 #define DEBUG
###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++-9 -m64
CXXFLAGS=-O3 -Wall
ifeq ($(ARCH), Darwin)
# Building on mac
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib/ -lcudart
else
# Building on Linux
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
endif
NVCC=nvcc
NVCCFLAGS=-O3 -m64

# 如果命令行指定了 debug=1，则启用调试标志 -DDEBUG
ifeq ($(debug), 1)
	NVCCFLAGS += $(DEBUG)
endif

OBJS=$(OBJDIR)/main.o  $(OBJDIR)/saxpy.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
