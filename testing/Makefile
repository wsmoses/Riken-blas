# Makefile
include ../make.inc

# CXX = ~/git/Polygeist/mlir-build/bin/clang++ --cuda-gpu-arch=sm_60 -resource-dir ~/git/Polygeist/mlir-build/lib/clang/14.0.0/ -L /usr/local/cuda-11.2/lib64 -l cudart_static -l dl -l rt -l pthread -l m -fopenmp -I /usr/local/cuda-11.2/targets/x86_64-linux/include -I /opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/include -I /home/wmoses/git/Polygeist/mlir-build/projects/openmp/runtime/src

# common -------------------------------------
FLAGS	= -O2 
# -Wall -MD -Wno-unused-but-set-variable
INCS	+= -I $(INC_PATH) -I ../include
LDFLAGS += -L $(LIB_PATH) -L ../lib
LINKAGE = static

# -------------------------------------------
ifeq ($(CUDA),yes)
OBJS	+= $(CUOZBLAS_OBJS) 
OBJS	+= $(CUBLAS_OBJS) 
INCS	+= -I $(CUDA_PATH)/include 
LDFLAGS	+= -L $(CUDA_PATH)/lib64
LIBS	+= -l cublasLt -l cudart -l cublas -l cusparse -l cuozblas
endif
#ifeq ($(CUBLAS),yes)
ifeq (no,yes)
INCS	+= -I $(CUDA_PATH)/include 
LDFLAGS	+= -L $(CUDA_PATH)/lib64 
LIBS	+= -l cudart -l cublas
FLAGS	+= -D CUBLAS
endif

ifeq (yes,yes)
OBJS	+= $(OZBLAS_OBJS) 
LIBS	+= -l ozblas 
ifeq ($(CPU_BLAS),MKL)
FLAGS	+= -D MKL
ifeq ($(LINKAGE),static)
LIBS	+= -Wl,--start-group $(MKLROOT)//libmkl_intel_lp64.a $(MKLROOT)/libmkl_gnu_thread.a $(MKLROOT)/libmkl_core.a -Wl,--end-group 
else
LIBS	+= -l mkl_intel_lp64 -l mkl_intel_thread -l mkl_core #-liomp5
endif
endif
ifeq ($(CPU_BLAS),OpenBLAS)
FLAGS	+= -D OpenBLAS
ifeq ($(LINKAGE),static)
LIBS	+= -Wl,-Bstatic,-lopenblas
else
LIBS	+= -l openblas
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
LIBS	+= -l mpblas_mpfr -l mpfr
endif
LIBS	+= -l mpc -l gmp -std=c++11 -std=gnu++17
FLAGS	+= -D MPLAPACK
endif

#ifeq ($(LINKAGE),static)
#LIBS	+= -Wl,-Bdynamic
#endif
LIBS	+= -l stdc++ -l m -l c -l dl 

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
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D CUDA -D CUOZBLAS -D GEMM -D PREC_S_S
testing_cuozblas_sgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D CUDA -D CUOZBLAS -D GEMV -D PREC_S_S
testing_cuozblas_sdot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D CUDA -D CUOZBLAS -D DOT -D PREC_S_S

testing_cuozblas_sdgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D CUDA -D CUOZBLAS -D GEMM -D PREC_S_D
testing_cuozblas_sdgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D CUDA -D CUOZBLAS -D GEMV -D PREC_S_D
testing_cuozblas_sddot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D CUDA -D CUOZBLAS -D DOT -D PREC_S_D

testing_cuozblas_dsgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D CUDA -D CUOZBLAS -D GEMM -D PREC_D_S
testing_cuozblas_dsgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D CUDA -D CUOZBLAS -D GEMV -D PREC_D_S
testing_cuozblas_dsdot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D CUDA -D CUOZBLAS -D DOT -D PREC_D_S

testing_cuozblas_dgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D CUDA -D CUOZBLAS -D GEMM -D PREC_D_D
testing_cuozblas_dgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D CUDA -D CUOZBLAS -D GEMV -D PREC_D_D
testing_cuozblas_ddot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D CUDA -D CUOZBLAS -D DOT -D PREC_D_D

### OZBLAS ###################################################################################################
testing_ozblas_sgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D OZBLAS -D GEMM -D PREC_S_S
testing_ozblas_sgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D OZBLAS -D GEMV -D PREC_S_S
testing_ozblas_sdot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D OZBLAS -D DOT -D PREC_S_S

testing_ozblas_sdgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D OZBLAS -D GEMM -D PREC_S_D
testing_ozblas_sdgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D OZBLAS -D GEMV -D PREC_S_D
testing_ozblas_sddot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D OZBLAS -D DOT -D PREC_S_D

testing_ozblas_dsgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D OZBLAS -D GEMM -D PREC_D_S
testing_ozblas_dsgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D OZBLAS -D GEMV -D PREC_D_S
testing_ozblas_dsdot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D OZBLAS -D DOT -D PREC_D_S

testing_ozblas_dgemm: $(SRC_L3)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D OZBLAS -D GEMM -D PREC_D_D
testing_ozblas_dgemv: $(SRC_L2)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D OZBLAS -D GEMV -D PREC_D_D
testing_ozblas_ddot: $(SRC_L1B)
	$(CXX) $(FLAGS) $(INCS) $(LDFLAGS) $< -o $@ $(LIBS) -D OZBLAS -D DOT -D PREC_D_D

clean:
	$(RM) $(OBJS)
	$(RM) *.d 

-include *.d
.PHONY: all clean

