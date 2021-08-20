# Makefile
include ../make.inc

# common -------------------------------------
FLAGS	= -O2 -Wall -MD -Wno-unused-but-set-variable
INCS	+= -I$(INC_PATH) -I../include
LDFLAGS += -L$(LIB_PATH) -L../lib

# -------------------------------------------
ifeq ($(CUDA),yes)
OBJS	+= $(CUOZBLAS_OBJS) 
OBJS	+= $(CUBLAS_OBJS) 
INCS	+= -I$(CUDA_PATH)/include 
LDFLAGS	+= -L$(CUDA_PATH)/lib64
LIBS	+= -lcublasLt -lcudart -lcublas -lcusparse -lcuozblas
endif
ifeq ($(CUBLAS),yes)
INCS	+= -I$(CUDA_PATH)/include 
LDFLAGS	+= -L$(CUDA_PATH)/lib64 
LIBS	+= -lcudart -lcublas
FLAGS	+= -DCUBLAS
endif

ifeq ($(CPU),yes)
OBJS	+= $(OZBLAS_OBJS) 
LIBS	+= -lozblas 
ifeq ($(CPU_BLAS),MKL)
FLAGS	+= -DMKL
ifeq ($(LINKAGE),static)
LIBS	+= -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group 
else
LIBS	+= -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core #-liomp5
endif
endif
ifeq ($(CPU_BLAS),OpenBLAS)
FLAGS	+= -DOpenBLAS
ifeq ($(LINKAGE),static)
LIBS	+= -Wl,-Bstatic,-lopenblas
else
LIBS	+= -lopenblas
endif
endif
endif

#=========================================
# below must be here
ifeq ($(MPLAPACK),yes)
OBJS	+= $(MPBLAS_OBJS) 
ifeq ($(LINKAGE),static)
LIBS	+= -Wl,-Bstatic,-lmpblas_mpfr -lmpfr
else
LIBS	+= -lmpblas_mpfr -lmpfr
endif
LIBS	+= -lmpc -lgmp -std=c++11 -std=gnu++17
FLAGS	+= -DMPLAPACK
endif

ifeq ($(LINKAGE),static)
LIBS	+= -Wl,-Bdynamic
endif
LIBS	+= -lstdc++ -lm -lc -ldl 

# compiler -----------------------------------
ifeq ($(CXX),icpc)
FLAGS	+= -qopenmp
ifeq ($(CPU_ARCH),KNL)
	FLAGS += -xMIC-AVX512
endif
ifeq ($(CPU_ARCH),SKYLAKE)
	FLAGS += -xCORE-AVX2 -mtune=skylake-avx512 
endif
endif
ifeq ($(CXX),g++)
	FLAGS += -fopenmp 
ifeq ($(CPU_ARCH),SKYLAKE)
	FLAGS += -march=skylake
endif
endif
#=========================================

# DOT  ---------------------------------------
ifeq ($(DOT),yes)
ifeq ($(PREC_S_S),yes)
CUOZBLAS_OBJS += testing_cuozblas_sdot
OZBLAS_OBJS += testing_ozblas_sdot
CUBLAS_OBJS += testing_cublas_sdot
endif
ifeq ($(PREC_S_D),yes)
CUOZBLAS_OBJS += testing_cuozblas_sddot
OZBLAS_OBJS += testing_ozblas_sddot
endif
ifeq ($(PREC_D_S),yes)
CUOZBLAS_OBJS += testing_cuozblas_dsdot
OZBLAS_OBJS += testing_ozblas_dsdot
endif
ifeq ($(PREC_D_D),yes)
CUOZBLAS_OBJS += testing_cuozblas_ddot
OZBLAS_OBJS += testing_ozblas_ddot
CUBLAS_OBJS += testing_cublas_ddot
endif
endif

# GEMV ---------------------------------------
ifeq ($(GEMV),yes)
ifeq ($(PREC_S_S),yes)
CUOZBLAS_OBJS += testing_cuozblas_sgemv
OZBLAS_OBJS += testing_ozblas_sgemv
CUBLAS_OBJS += testing_cublas_sgemv
endif
ifeq ($(PREC_S_D),yes)
CUOZBLAS_OBJS += testing_cuozblas_sdgemv
OZBLAS_OBJS += testing_ozblas_sdgemv
endif
ifeq ($(PREC_D_S),yes)
CUOZBLAS_OBJS += testing_cuozblas_dsgemv
OZBLAS_OBJS += testing_ozblas_dsgemv
endif
ifeq ($(PREC_D_D),yes)
CUOZBLAS_OBJS += testing_cuozblas_dgemv
OZBLAS_OBJS += testing_ozblas_dgemv
CUBLAS_OBJS += testing_cublas_dgemv
endif
endif

# GEMM ---------------------------------------
ifeq ($(GEMM),yes)
ifeq ($(PREC_S_S),yes)
CUOZBLAS_OBJS += testing_cuozblas_sgemm
OZBLAS_OBJS += testing_ozblas_sgemm
CUBLAS_OBJS += testing_cublas_sgemm
endif
ifeq ($(PREC_S_D),yes)
CUOZBLAS_OBJS += testing_cuozblas_sdgemm
OZBLAS_OBJS += testing_ozblas_sdgemm
endif
ifeq ($(PREC_D_S),yes)
CUOZBLAS_OBJS += testing_cuozblas_dsgemm
OZBLAS_OBJS += testing_ozblas_dsgemm
endif
ifeq ($(PREC_D_D),yes)
CUOZBLAS_OBJS += testing_cuozblas_dgemm
OZBLAS_OBJS += testing_ozblas_dgemm
CUBLAS_OBJS += testing_cublas_dgemm
endif
endif

default: $(OBJS)
ifeq ($(CPU),yes)
$(OBJS): ../lib/libozblas.a
endif
ifeq ($(CUDA),yes)
$(OBJS): ../lib/libcuozblas.a
endif
$(OBJS): \
testing_common.cpp \
testing_common.h \
testing_setting.h

SRC_L1B	= testing_level1b.cpp 
SRC_L2	= testing_level2.cpp 
SRC_L3	= testing_level3.cpp 

### CUOZBLAS ###################################################################################################
testing_cuozblas_sgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DGEMM -DPREC_S_S
testing_cuozblas_sgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DGEMV -DPREC_S_S
testing_cuozblas_sdot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DDOT -DPREC_S_S

testing_cuozblas_sdgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DGEMM -DPREC_S_D
testing_cuozblas_sdgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DGEMV -DPREC_S_D
testing_cuozblas_sddot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DDOT -DPREC_S_D

testing_cuozblas_dsgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DGEMM -DPREC_D_S
testing_cuozblas_dsgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DGEMV -DPREC_D_S
testing_cuozblas_dsdot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DDOT -DPREC_D_S

testing_cuozblas_dgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DGEMM -DPREC_D_D
testing_cuozblas_dgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DGEMV -DPREC_D_D
testing_cuozblas_ddot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DCUDA -DCUOZBLAS -DDOT -DPREC_D_D

### OZBLAS ###################################################################################################
testing_ozblas_sgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMM -DPREC_S_S
testing_ozblas_sgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMV -DPREC_S_S
testing_ozblas_sdot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DDOT -DPREC_S_S

testing_ozblas_sdgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMM -DPREC_S_D
testing_ozblas_sdgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMV -DPREC_S_D
testing_ozblas_sddot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DDOT -DPREC_S_D

testing_ozblas_dsgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMM -DPREC_D_S
testing_ozblas_dsgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMV -DPREC_D_S
testing_ozblas_dsdot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DDOT -DPREC_D_S

testing_ozblas_dgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMM -DPREC_D_D
testing_ozblas_dgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DGEMV -DPREC_D_D
testing_ozblas_ddot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -DOZBLAS -DDOT -DPREC_D_D

clean:
	$(RM) $(OBJS)
	$(RM) *.d 

-include *.d
.PHONY: all clean

