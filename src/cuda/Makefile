# Makefile
include ../../make.inc

NVCC	= $(CXX)
AR		= ar
LIB		:= ../../lib/libcuozblas.a
FLAGS   := -O3 -I $(INC_PATH) 
NVCCFLAGS   := -O3 

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
	$(CXX) $(FLAGS) -o $@ -c $< -I $(CUDA_PATH)/include
#cuozblas.o: cuozblas.cpp
	$(CXX) $(FLAGS) -o $@ -c $< -I $(CUDA_PATH)/include

clean :
	$(RM) *.o *.d
	$(RM) $(LIB)
