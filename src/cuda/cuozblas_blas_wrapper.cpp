#include "cuozblas_common.h"

// =========================================
// BLAS Wrappers
// =========================================
// IAMAX
void blasRiamax (cublasHandle_t ch, const int32_t n, const float* x, const int32_t incx, int32_t* ret) {
	//cublasIsamax (ch, n, x, incx, ret);
}
void blasRiamax (cublasHandle_t ch, const int32_t n, const double* x, const int32_t incx, int32_t* ret) {
	//cublasIdamax (ch, n, x, incx, ret);
}
// ASUM
void blasRasum (cublasHandle_t ch, const int32_t n, const float* x, const int32_t incx, float* ret) {
	//cublasSasum (ch, n, x, incx, ret);
}
void blasRasum (cublasHandle_t ch, const int32_t n, const double* x, const int32_t incx, double* ret) {
	//cublasDasum (ch, n, x, incx, ret);
}
// DOT
void blasRdot (cublasHandle_t ch, const int32_t n, const float* x, const int32_t incx, const float* y, const int32_t incy, float* ret) {
	//cublasSdot (ch, n, x, incx, y, incy, ret);
}
void blasRdot (cublasHandle_t ch, const int32_t n, const double* x, const int32_t incx, const double* y, const int32_t incy, double* ret) {
	//cublasDdot (ch, n, x, incx, y, incy, ret);
}
// GEMV
void blasRgemv (cublasHandle_t ch, const char trans, const int32_t m, const int32_t n, const float alpha, const float* A, const int32_t lda, const float* x, const int32_t incx, const float beta, float* y, const int32_t incy) {
	//cublasSgemv (ch, ToCublasOp(trans), m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}
void blasRgemv (cublasHandle_t ch, const char trans, const int32_t m, const int32_t n, const double alpha, const double* A, const int32_t lda, const double* x, const int32_t incx, const double beta, double* y, const int32_t incy) {
	//cublasDgemv (ch, ToCublasOp(trans), m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}
// GEMM
void blasRgemm (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float* A, const int32_t lda, const float* B, const int32_t ldb, const float beta, float* C, const int32_t ldc) {
	//cublasSgemm (ch, ToCublasOp(transA), ToCublasOp(transB), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}
void blasRgemm (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double* A, const int32_t lda, const double* B, const int32_t ldb, const double beta, double* C, const int32_t ldc) {
	//cublasDgemm (ch, ToCublasOp(transA), ToCublasOp(transB), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}
// GEMM-BATCH
void blasRgemmBatch (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float** A, const int32_t lda, const float** B, const int32_t ldb, const float beta, float** C, const int32_t ldc, const int32_t grp, const int32_t cnt) {
	//cublasSgemmBatched (ch, ToCublasOp(transA), ToCublasOp(transB), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc, cnt);
}
void blasRgemmBatch (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double** A, const int32_t lda, const double** B, const int32_t ldb, const double beta, double** C, const int32_t ldc, const int32_t grp, const int32_t cnt) {
	//cublasDgemmBatched (ch, ToCublasOp(transA), ToCublasOp(transB), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc, cnt);
}
