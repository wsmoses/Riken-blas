# Makefile
include ../../make.inc

NVCC	= $(CUDA_PATH)/bin/nvcc 
AR		= ar
LIB		:= ../../lib/libcuozblas.a
FLAGS   := -Wall -MD -O3 -Wno-maybe-uninitialized -I$(INC_PATH) 
NVCCFLAGS   := -O3 --expt-relaxed-constexpr

ifeq ($(CC60),yes)
NVCCFLAGS += -gencode arch=compute_60,code=sm_60
endif
ifeq ($(CC61),yes)
NVCCFLAGS += -gencode arch=compute_61,code=sm_61
endif
ifeq ($(CC70),yes)
NVCCFLAGS += -gencode arch=compute_60,code=sm_70
endif
ifeq ($(CC75),yes)
NVCCFLAGS += -gencode arch=compute_60,code=sm_75
endif
ifeq ($(CC86),yes)
NVCCFLAGS += -gencode arch=compute_60,code=sm_86
endif

all: blas

ALLSRC += \
cuozblas_blas_wrapper.cpp \
cuozblas_aux.cpp \
cuozblas_split.cu \
cuozblas_sum.cu 

ALLHDR = \
cuozblas_common.h \
cuozblas_internal.h 

ALLSRC += cuozblas_gemm.cpp
ifeq ($(DOT),yes)
ALLSRC += cuozblas_dot.cpp
endif
ifeq ($(GEMV),yes)
ALLSRC += cuozblas_gemv.cpp
endif

ALLOBJ = cuozblas.o
ALLOBJ += $(ALLSRC:.cu=.o)
ALLOBJ += $(ALLSRC:.cpp=.o)

$(ALLOBJ) : $(ALLHDR)

blas: $(ALLOBJ) 
	$(AR) rv $(LIB) $(ALLOBJ) 

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $< 
%.o: %.cpp
	$(CXX) $(FLAGS) -o $@ -c $< -I$(CUDA_PATH)/include
#cuozblas.o: cuozblas.cpp
	$(CXX) $(FLAGS) -o $@ -c $< -I$(CUDA_PATH)/include

clean :
	$(RM) *.o *.d
	$(RM) $(LIB)
