##############################################

#--- System ----------------------------------
#CPU_ARCH: KNL, SKYLAKE
#CPU_ARCH = SKYLAKE
# -> If specified, compile with architecture-specific options

#LINKAGE: static, dynamic
LINKAGE = static

#--- PATH ------------------------------------
CUDA_PATH	= /usr/local/cuda-11.4
LIB_PATH	= $(HOME)/local/lib
INC_PATH	= $(HOME)/local/include
# -> installation path of external libraries

#--- Compiler --------------------------------
#CXX: g++, icpc
CXX = ~/git/Polygeist/mlir-build/bin/clang++ --cuda-gpu-arch=sm_60 -L /usr/local/cuda-11.2/lib64 -lcudart_static -ldl -lrt -lpthread -lm -fopenmp -I /usr/local/cuda-11.2/targets/x86_64-linux/include

#--- CPU (OzBLAS) / CUDA (cuOzBLAS) ----------
CPU  = yes
CUDA = yes

#--- BLAS (CPU) ------------------------------
CPU_BLAS = MKL

#--- MPLAPACK (for testing) ------------------
MPLAPACK = no
# -> yes or no. 'yes' uses MPLAPACK for verifying computational result
# -> If no, binary128 (PREC_Q_D and PREC_Q_S) cannot be used.

#--- CUDA Compute Capability -----------------
CC60 = no
CC61 = no
CC62 = no
CC70 = yes
CC75 = no
CC80 = no
CC86 = yes
# -> Tested only on GPUs with CC70, CC75, and CC86

#--- Precision -------------------------------
# PREC_X_Y: X is I/O-precision, Y is compute-precision
# Q: FP128, D: FP64, S: FP32, H: FP16, B: BF16
PREC_D_D = yes
PREC_D_S = no
PREC_S_S = no
PREC_S_D = no

#--- Routine ---------------------------------
DOT = yes
GEMV = yes
GEMM = yes

##############################################

