#include "ozblas_common.h"

// =========================================
// BLAS Wrappers
// =========================================
// IAMAX
int32_t blasRiamax (const int32_t n, const float* x, const int32_t incx) {
	return cblas_isamax (n, x, incx);
}
int32_t blasRiamax (const int32_t n, const double* x, const int32_t incx) {
	return cblas_idamax (n, x, incx);
}
#if defined (FLOAT128)
int32_t blasRiamax (const int32_t n, const __float128* x, const int32_t incx) {
	#if defined (MPLAPACK)
	return iRamax (n, (__float128*)x, incx) - 1; // MPLAPACK uses 1-based index
	#else
	fprintf (OUTPUT, "OzBLAS error: iRamax (binary128) is not available. Use MPLAPACK.\n");
	exit(1);
	#endif
}
#endif
// ASUM
float blasRasum (const int32_t n, const float* x, const int32_t incx) {
	return cblas_sasum (n, x, incx);
}
double blasRasum (const int32_t n, const double* x, const int32_t incx) {
	return cblas_dasum (n, x, incx);
}
#if defined (FLOAT128)
__float128 blasRasum (const int32_t n, const __float128* x, const int32_t incx) {
	#if defined (MPLAPACK)
	return Rasum (n, (__float128*)x, incx);
	#else
	fprintf (OUTPUT, "OzBLAS error: Rasum (binary128) is not available. Use MPLAPACK.\n");
	exit(1);
	#endif
}
#endif
// SCAL
void blasRscal (const int32_t n, const float alpha, float* x, const int32_t incx) {
	cblas_sscal (n, alpha, x, incx);
}
void blasRscal (const int32_t n, const double alpha, double* x, const int32_t incx) {
	cblas_dscal (n, alpha, x, incx);
}
#if defined (FLOAT128)
void blasRscal (const int32_t n, const __float128 alpha, __float128* x, const int32_t incx) {
	#if defined (MPLAPACK)
	Rscal (n, alpha, (__float128*)x, incx);
	#else
	fprintf (OUTPUT, "OzBLAS error: Rscal (binary128) is not available. Use MPLAPACK.\n");
	exit(1);
	#endif
}
#endif
// AXPY
void blasRaxpy (const int32_t n, const float alpha, const float* x, const int32_t incx, float* y, const int32_t incy) {
	cblas_saxpy (n, alpha, x, incx, y, incy);
}
void blasRaxpy (const int32_t n, const double alpha, const double* x, const int32_t incx, double* y, const int32_t incy) {
	cblas_daxpy (n, alpha, x, incx, y, incy);
}
#if defined (FLOAT128)
void blasRaxpy (const int32_t n, const __float128 alpha, const __float128* x, const int32_t incx, __float128* y, const int32_t incy) {
	#if defined (MPLAPACK)
	Raxpy (n, alpha, (__float128*)x, incx, (__float128*)y, incy);
	#else
	fprintf (OUTPUT, "OzBLAS error: Raxpy (binary128) is not available. Use MPLAPACK.\n");
	exit(1);
	#endif
}
#endif
// DOT
float blasRdot (const int32_t n, const float* x, const int32_t incx, const float* y, const int32_t incy) {
	return cblas_sdot (n, x, incx, y, incy);
}
double blasRdot (const int32_t n, const double* x, const int32_t incx, const double* y, const int32_t incy) {
	return cblas_ddot (n, x, incx, y, incy);
}
#if defined (FLOAT128)
__float128 blasRdot (const int32_t n, const __float128* x, const int32_t incx, const __float128* y, const int32_t incy) {
	#if defined (MPLAPACK)
	return Rdot (n, (__float128*)x, incx, (__float128*)y, incy);
	#else
	fprintf (OUTPUT, "OzBLAS error: Rdot (binary128) is not available. Use MPLAPACK.\n");
	exit(1);
	#endif
}
#endif
// NRM2
float blasRnrm2 (const int32_t n, const float* x, const int32_t incx) {
	return cblas_snrm2 (n, x, incx);
}
double blasRnrm2 (const int32_t n, const double* x, const int32_t incx) {
	return cblas_dnrm2 (n, x, incx);
}
#if defined (FLOAT128)
__float128 blasRnrm2 (const int32_t n, const __float128* x, const int32_t incx) {
	#if defined (MPLAPACK)
	return Rnrm2 (n, (__float128*)x, incx);
	#else
	fprintf (OUTPUT, "OzBLAS error: Rnrm2 (binary128) is not available. Use MPLAPACK.\n");
	exit(1);
	#endif
}
#endif
// GEMV
void blasRgemv (const char trans, const int32_t m, const int32_t n, const float alpha, const float* A, const int32_t lda, const float* x, const int32_t incx, const float beta, float* y, const int32_t incy) {
	cblas_sgemv (CblasColMajor, ToCblasOp(trans), m, n, alpha, A, lda, x, incx, beta, y, incy);
}
void blasRgemv (const char trans, const int32_t m, const int32_t n, const double alpha, const double* A, const int32_t lda, const double* x, const int32_t incx, const double beta, double* y, const int32_t incy) {
	cblas_dgemv (CblasColMajor, ToCblasOp(trans), m, n, alpha, A, lda, x, incx, beta, y, incy);
}
#if defined (FLOAT128)
void blasRgemv (const char trans, const int32_t m, const int32_t n, const __float128 alpha, const __float128* A, const int32_t lda, const __float128* x, const int32_t incx, const __float128 beta, __float128* y, const int32_t incy) {
	#if defined (MPLAPACK)
	Rgemv (&trans, m, n, alpha, (__float128*)A, lda, (__float128*)x, incx, beta, y, incy);
	#else
	fprintf (OUTPUT, "OzBLAS error: Rgemv (binary128) is not available. Use MPLAPACK.\n");
	exit(1);
	#endif
}
#endif
// GEMM
void blasRgemm (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float* A, const int32_t lda, const float* B, const int32_t ldb, const float beta, float* C, const int32_t ldc) {
	cblas_sgemm (CblasColMajor, ToCblasOp(transA), ToCblasOp(transB), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
void blasRgemm (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double* A, const int32_t lda, const double* B, const int32_t ldb, const double beta, double* C, const int32_t ldc) {
	cblas_dgemm (CblasColMajor, ToCblasOp(transA), ToCblasOp(transB), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
#if defined (FLOAT128)
void blasRgemm (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const __float128 alpha, const __float128* A, const int32_t lda, const __float128* B, const int32_t ldb, const __float128 beta, __float128* C, const int32_t ldc) {
	#if defined (MPLAPACK)
	Rgemm (&transA, &transB, m, n, k, alpha, (__float128*)A, lda, (__float128*)B, ldb, beta, C, ldc);
	#else
	fprintf (OUTPUT, "OzBLAS error: Rgemm (binary128) is not available. Use MPLAPACK.\n");
	exit(1);
	#endif
}
#endif
// CUDA-GEMM
#if defined (CUBLAS)
void cublasRgemm (cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, const int32_t m, const int32_t n, const int32_t k, const float* alpha, const float* A, const int32_t lda, const float* B, const int32_t ldb, const float* beta, float* C, const int32_t ldc) {
	cublasSgemm (handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
void cublasRgemm (cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, const int32_t m, const int32_t n, const int32_t k, const double* alpha, const double* A, const int32_t lda, const double* B, const int32_t ldb, const double* beta, double* C, const int32_t ldc) {
	cublasDgemm (handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
#endif

// GEMM-BATCH
void blasRgemmBatch (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float** A, const int32_t lda, const float** B, const int32_t ldb, const float beta, float** C, const int32_t ldc, const int32_t grp, const int32_t cnt) {
	#if defined (MKL)
	CBLAS_TRANSPOSE transA_ = ToCblasOp (transA);
	CBLAS_TRANSPOSE transB_ = ToCblasOp (transB);
	cblas_sgemm_batch (CblasColMajor, &transA_, &transB_, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, grp, &cnt);
	#else
	fprintf (OUTPUT, "OzBLAS error: GEMM_BATCH is not available.\n");
	exit(1);
	#endif
}
void blasRgemmBatch (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double** A, const int32_t lda, const double** B, const int32_t ldb, const double beta, double** C, const int32_t ldc, const int32_t grp, const int32_t cnt) {
	#if defined (MKL)
	CBLAS_TRANSPOSE transA_ = ToCblasOp (transA);
	CBLAS_TRANSPOSE transB_ = ToCblasOp (transB);
	cblas_dgemm_batch (CblasColMajor, &transA_, &transB_, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, grp, &cnt);
	#else
	fprintf (OUTPUT, "OzBLAS error: GEMM_BATCH is not available.\n");
	exit(1);
	#endif
}
// CSRMV
template <typename TYPE>
void stdCsrmv (
	const int32_t m, const int32_t n,
	const TYPE alpha, const TYPE* matA, const int32_t* matAind, const int32_t* matAptr,
	const TYPE* x, const TYPE beta, TYPE* y
	) {
	#pragma omp parallel for
	for(int32_t j = 0; j < m; j++) {
		TYPE t = 0.;
		for(int32_t i = matAptr[j]; i < matAptr[j+1]; i++) 
			t = t + matA[i] * x[matAind[i]];
		y[j] = alpha * t + beta * y[j];
	}
}
#if defined (FLOAT128)
template void stdCsrmv (const int32_t m, const int32_t n, const __float128 alpha, const __float128* matA, const int32_t* matAind, const int32_t* matAptr, const __float128* x, const __float128 beta, __float128* y);
#endif
template void stdCsrmv (const int32_t m, const int32_t n, const double alpha, const double* matA, const int32_t* matAind, const int32_t* matAptr, const double* x, const double beta, double* y);
template void stdCsrmv (const int32_t m, const int32_t n, const float alpha, const float* matA, const int32_t* matAind, const int32_t* matAptr, const float* x, const float beta, float* y);

void blasRcsrmv (const char trans, const int32_t m, const int32_t n, const float alpha, const char *descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *X, const float beta, float *Y) {
	#if defined (MKL)
// --- with Inspector-Executor Sparse BLAS ---
	int32_t expected_calls = 1;
	struct matrix_descr descrA_matrix;
	sparse_matrix_t csrA;
	mkl_sparse_s_create_csr (&csrA, SPARSE_INDEX_BASE_ZERO, m, n, (int32_t*)devArowptr, (int32_t*)devArowptr+1, (int32_t*)devAcolind, (float*)A);
	descrA_matrix.type = SPARSE_MATRIX_TYPE_GENERAL;//SPARSE_MATRIX_TYPE_SYMMETRIC;
//	descrA_matrix.mode = SPARSE_FILL_MODE_UPPER;
//	descrA_matrix.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_set_mv_hint (csrA, SPARSE_OPERATION_NON_TRANSPOSE, descrA_matrix, expected_calls);
	mkl_sparse_s_mv (SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descrA_matrix, X, beta, Y);
	mkl_sparse_destroy (csrA);
// ------------------------------------------
//	mkl_scsrmv (&trans, &m, &n, &alpha, descrA, A, devAcolind, devArowptr, devArowptr+1, X, &beta, Y);
	#else
	fprintf (OUTPUT, "OzBLAS warning: in-house CSRMV is used.\n");
	stdCsrmv (m, n, alpha, A, devAcolind, devArowptr, X, beta, Y);
	#endif
}
void blasRcsrmv (const char trans, const int32_t m, const int32_t n, const double alpha, const char *descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *X, const double beta, double *Y) {
	#if defined (MKL)
// --- with Inspector-Executor Sparse BLAS ---
	int32_t expected_calls = 1;
	struct matrix_descr descrA_matrix;
	sparse_matrix_t csrA;
	mkl_sparse_d_create_csr (&csrA, SPARSE_INDEX_BASE_ZERO, m, n, (int32_t*)devArowptr, (int32_t*)devArowptr+1, (int32_t*)devAcolind, (double*)A);
	descrA_matrix.type = SPARSE_MATRIX_TYPE_GENERAL;//SPARSE_MATRIX_TYPE_SYMMETRIC;
//	descrA_matrix.mode = SPARSE_FILL_MODE_UPPER;
//	descrA_matrix.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_set_mv_hint (csrA, SPARSE_OPERATION_NON_TRANSPOSE, descrA_matrix, expected_calls);
	mkl_sparse_d_mv (SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descrA_matrix, X, beta, Y);
	mkl_sparse_destroy (csrA);
// ------------------------------------------
//	mkl_dcsrmv (&trans, &m, &n, &alpha, descrA, A, devAcolind, devArowptr, devArowptr+1, X, &beta, Y);
	#else
	fprintf (OUTPUT, "OzBLAS warning: in-house CSRMV is used.\n");
	stdCsrmv (m, n, alpha, A, devAcolind, devArowptr, X, beta, Y);
	#endif
}
#if defined (FLOAT128)
void blasRcsrmv (const char trans, const int32_t m, const int32_t n, const __float128 alpha, const char *descrA, const __float128 *A, const int32_t *devAcolind, const int32_t *devArowptr, const __float128 *X, const __float128 beta, __float128 *Y) {
	stdCsrmv (m, n, alpha, A, devAcolind, devArowptr, X, beta, Y);
}
#endif
// CSRMM
void blasRcsrmm (const char trans, const int32_t m, const int32_t n, const int32_t k, const float alpha, const char *descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *B, const int32_t ldb, const float beta, float *C, const int32_t ldc) {
	#if defined (MKL)
// --- with Inspector-Executor Sparse BLAS ---
	int32_t expected_calls = 1;
	struct matrix_descr descrA_matrix;
	sparse_matrix_t csrA;
	mkl_sparse_s_create_csr (&csrA, SPARSE_INDEX_BASE_ZERO, m, k, (int32_t*)devArowptr, (int32_t*)devArowptr+1, (int32_t*)devAcolind, (float*)A);
	descrA_matrix.type = SPARSE_MATRIX_TYPE_GENERAL;//SPARSE_MATRIX_TYPE_SYMMETRIC;
//	descrA_matrix.mode = SPARSE_FILL_MODE_UPPER;
//	descrA_matrix.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_set_mm_hint (csrA, SPARSE_OPERATION_NON_TRANSPOSE, descrA_matrix, SPARSE_LAYOUT_COLUMN_MAJOR, n, expected_calls);
	mkl_sparse_s_mm (SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descrA_matrix, SPARSE_LAYOUT_COLUMN_MAJOR, B, n, ldb, beta, C, ldc);
	mkl_sparse_destroy (csrA);
// ------------------------------------------
//	mkl_scsrmm (&trans, &m, &n, &k, &alpha, descrA, A, devAcolind, devArowptr, devArowptr+1, B, &ldb, &beta, C, &ldc);
	#else
	fprintf (OUTPUT, "OzBLAS error: CSRMM is not available.\n");
	exit(1);
	#endif
}
void blasRcsrmm (const char trans, const int32_t m, const int32_t n, const int32_t k, const double alpha, const char *descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *B, const int32_t ldb, const double beta, double *C, const int32_t ldc) {
	#if defined (MKL)
// --- with Inspector-Executor Sparse BLAS ---
	int32_t expected_calls = 1;
	struct matrix_descr descrA_matrix;
	sparse_matrix_t csrA;
	mkl_sparse_d_create_csr (&csrA, SPARSE_INDEX_BASE_ZERO, m, k, (int32_t*)devArowptr, (int32_t*)devArowptr+1, (int32_t*)devAcolind, (double*)A);
	descrA_matrix.type = SPARSE_MATRIX_TYPE_GENERAL;//SPARSE_MATRIX_TYPE_SYMMETRIC;
//	descrA_matrix.mode = SPARSE_FILL_MODE_UPPER;
//	descrA_matrix.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_set_mm_hint (csrA, SPARSE_OPERATION_NON_TRANSPOSE, descrA_matrix, SPARSE_LAYOUT_COLUMN_MAJOR, n, expected_calls);
	mkl_sparse_d_mm (SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descrA_matrix, SPARSE_LAYOUT_COLUMN_MAJOR, B, n, ldb, beta, C, ldc);
	mkl_sparse_destroy (csrA);
// ------------------------------------------
//	mkl_dcsrmm (&trans, &m, &n, &k, &alpha, descrA, A, devAcolind, devArowptr, devArowptr+1, B, &ldb, &beta, C, &ldc);
	#else
	fprintf (OUTPUT, "OzBLAS error: CSRMM is not available.\n");
	exit(1);
	#endif
}
// OMATCOPY
void blasRomatcopy (const char trans, const int32_t m, const int32_t n, const float* A, const int32_t lda, float* B, const int32_t ldb) {
	#if defined (MKL)
	mkl_somatcopy ('c', trans, m, n, 1., A, lda, B, ldb);
	#elif defined (SSL2)
	fprintf (OUTPUT, "OzBLAS error: omatcopy is not available.\n");
	exit(1);
	#else
	cblas_somatcopy (CblasColMajor, ToCblasOp(trans), m, n, 1., A, lda, B, ldb);
	#endif
}
void blasRomatcopy (const char trans, const int32_t m, const int32_t n, const double* A, const int32_t lda, double* B, const int32_t ldb) {
	#if defined (MKL)
	mkl_domatcopy ('c', trans, m, n, 1., A, lda, B, ldb);
	#elif defined (SSL2)
	fprintf (OUTPUT, "OzBLAS error: omatcopy is not available.\n");
	exit(1);
	#else
	cblas_domatcopy (CblasColMajor, ToCblasOp(trans), m, n, 1., A, lda, B, ldb);
	#endif
}
#if defined (FLOAT128)
#include <complex>
void blasRomatcopy (const char trans, const int32_t m, const int32_t n, const __float128* A, const int32_t lda, __float128* B, const int32_t ldb) {
	#if defined (MKL)
	MKL_Complex16 zone;
	zone.real = 1.;
	zone.imag = 0.;
	mkl_zomatcopy ('c', trans, m, n, zone, (MKL_Complex16*)A, lda, (MKL_Complex16*)B, ldb);
	#elif defined (SSL2)
	fprintf (OUTPUT, "OzBLAS error: omatcopy is not available.\n");
	exit(1);
	#else
	const double done = 1.;
	cblas_zomatcopy (CblasColMajor, ToCblasOp(trans), m, n, &done, (const double*)A, lda, (double*)B, ldb);
	#endif
}
#endif
