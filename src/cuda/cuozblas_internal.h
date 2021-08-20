#ifndef CUOZBLAS_INTERNAL_H
#define CUOZBLAS_INTERNAL_H

//=============================================
// cuozblas_aux.cpp
//=============================================
int32_t cucheckTrans (const char tran);
double cutimer ();
int32_t cumemCheck (cuozblasHandle_t *oh);
void cuozblasMatAddrAlloc (cuozblasHandle_t *oh, const int32_t m, const int32_t n, const int32_t size, void **dev, int32_t &lds);
void cuozblasVecAddrAlloc (cuozblasHandle_t *oh, const int32_t m, const int32_t size, void **dev);
cublasOperation_t ToCublasOp (const char tran);

//=============================================
// cuozblas_XXX
//=============================================
int32_t cugetPitchSize (int32_t n);
void cucounterInit (cuozblasHandle_t *oh);
template <typename TYPE1, typename TYPE2> int32_t cuozblasSplit (cuozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const TYPE1 *devInput, const int32_t ldi, TYPE1 *devOutput, const int32_t ldo, TYPE2 *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, TYPE1 *devMax);
template <typename TYPE1, typename TYPE2> int32_t cuozblasGlobalSum (cuozblasHandle_t *oh, const int32_t m, const int32_t n, const short *devASpExp, const int32_t ldas, const int32_t numsplitA, const short *devBSpExp, const int32_t ldbs, const int32_t numsplitB, TYPE2 *devCsplit, const int32_t llsc, const int32_t ldsc, TYPE1 *devC, const int32_t ldc, const TYPE1 alpha, const TYPE1 beta, const int32_t maxlevel, const int32_t sumOrder);

//=============================================
// BLAS Wrapper
//=============================================
// IAMAX
void blasRiamax (cublasHandle_t ch, const int32_t n, const float* x, const int32_t incx, int32_t* ret);
void blasRiamax (cublasHandle_t ch, const int32_t n, const double* x, const int32_t incx, int32_t* ret);
// ASUM
void blasRasum (cublasHandle_t ch, const int32_t n, const float* x, const int32_t incx, float* ret);
void blasRasum (cublasHandle_t ch, const int32_t n, const double* x, const int32_t incx, double* ret);
// DOT
void blasRdot (cublasHandle_t ch, const int32_t n, const float* x, const int32_t incx, const float* y, const int32_t incy, float* ret);
void blasRdot (cublasHandle_t ch, const int32_t n, const double* x, const int32_t incx, const double* y, const int32_t incy, double* ret);
// GEMV
void blasRgemv (cublasHandle_t ch, const char trans, const int32_t m, const int32_t n, const float alpha, const float* A, const int32_t lda, const float* x, const int32_t incx, const float beta, float* y, const int32_t incy);
void blasRgemv (cublasHandle_t ch, const char trans, const int32_t m, const int32_t n, const double alpha, const double* A, const int32_t lda, const double* x, const int32_t incx, const double beta, double* y, const int32_t incy);
// GEMM
void blasRgemm (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float* A, const int32_t lda, const float* B, const int32_t ldb, const float beta, float* C, const int32_t ldc);
void blasRgemm (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double* A, const int32_t lda, const double* B, const int32_t ldb, const double beta, double* C, const int32_t ldc);
// GEMM-BATCH
void blasRgemmBatch (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float** A, const int32_t lda, const float** B, const int32_t ldb, const float beta, float** C, const int32_t ldc, const int32_t grp, const int32_t cnt);
void blasRgemmBatch (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double** A, const int32_t lda, const double** B, const int32_t ldb, const double beta, double** C, const int32_t ldc, const int32_t grp, const int32_t cnt);

#endif
