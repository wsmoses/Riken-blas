#ifndef CUOZBLAS_H
#define CUOZBLAS_H

#include <cublas_v2.h>

typedef struct {
	cublasHandle_t ch;

	// core 
	char* devWork;
	char* devWorkCommon;
	uint64_t workSizeBytes;
	uint64_t memAddr;
	int32_t nSplitMax;
	char* hstBatchAddr;
	char* devBatchAddr;

	int32_t initialized;

	// option flags
	int32_t splitModeFlag;
	int32_t fastModeFlag;
	int32_t reproModeFlag;
	int32_t sumModeFlag;
	int32_t useBatchedGemmFlag;
	int32_t overflowModeFlag;

	// exec info
	float nSplitA;
	float nSplitB;
	float nSplitC;

	float t_SplitA;
	float t_SplitB;
	float t_comp;
	float t_sum;
	float t_total;
	float n_comp;
	int32_t mbk;
	int32_t nbk;

	// for SpMV in iterative solvers
	int32_t splitShift;
	uint64_t memMaskSplitA;
} cuozblasHandle_t;

// helper routines
extern void cuozblasCreate (cuozblasHandle_t*, uint64_t);
extern void cuozblasDestroy (cuozblasHandle_t*);

// ================================
// BLAS template
// ================================
template <typename TYPE1, typename TYPE2> int32_t cuozblasRdot (cuozblasHandle_t *oh, const int32_t n, const TYPE1 *devA, const int32_t incx, const TYPE1 *devB, const int32_t incy, TYPE1 *ret);
template <typename TYPE1, typename TYPE2> int32_t cuozblasRgemv (cuozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const TYPE1 alpha, const TYPE1 *devA, const int32_t lda, const TYPE1 *devB, const int32_t incx, const TYPE1 beta, TYPE1 *devC, const int32_t incy);
template <typename TYPE1, typename TYPE2> int32_t cuozblasRgemm (cuozblasHandle_t *oh,	const char tranA, const char tranB, const int32_t m, const int32_t n, const int32_t k, const TYPE1 alpha, const TYPE1 *devA, const int32_t lda, const TYPE1 *devB, const int32_t ldb, const TYPE1 beta, TYPE1 *devC, const int32_t ldc);
#endif
