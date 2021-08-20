#include "cuozblas_common.h"

int32_t cucheckTrans (const char tran) {
	if (tran == 'N' || tran == 'n') 
		return 0;
	else
		return 1;
}

cublasOperation_t ToCublasOp (const char tran) {
	if (tran == 'N' || tran == 'n') return CUBLAS_OP_N;
	if (tran == 'T' || tran == 't') return CUBLAS_OP_T;
	if (tran == 'C' || tran == 'c') return CUBLAS_OP_C;
	return CUBLAS_OP_N; //default
}

// =========================================
// Matrix Allocation
// =========================================

int32_t cumemCheck (cuozblasHandle_t *oh) {
	if (oh->memAddr > oh->workSizeBytes) return 1;
	return 0;
}

void cuozblasMatAddrAlloc (
	cuozblasHandle_t *oh,
	const int32_t m,
	const int32_t n,
	const int32_t size,
	void **dev,
	int32_t &ld
) {
	ld = cugetPitchSize (m);
	dev[0] = oh->devWork + oh->memAddr;
	oh->memAddr += (uint64_t)size * ld * n;
}

void cuozblasVecAddrAlloc (
	cuozblasHandle_t *oh,
	const int32_t n,
	const int32_t size,
	void **dev
) {
	int32_t ld = cugetPitchSize (n);
	dev[0] = oh->devWork + oh->memAddr;
	oh->memAddr += (uint64_t)size * ld;
}

double cutimer () {
	struct timeval tv;
	cudaDeviceSynchronize ();
	gettimeofday (&tv, NULL);
	return tv.tv_sec + (double) tv.tv_usec * 1.0e-6;
}

// note: this is temporal...
int32_t cugetPitchSize (int32_t n) {
	return ceil((float)n / 128) * 128;
}

