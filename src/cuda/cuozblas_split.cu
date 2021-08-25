#include "cuozblas_common.h"

// =========================================
// Split
// Based on K.Ozaki et al., "Error-free transformations of matrix multiplication
// by using fast routines of matrix multiplication and its applications", 2012
// =========================================

// Number of threads and blocks for CUDA kernels
#define SPLIT_N_NTX 32
#define SPLIT_N_NTY 16
#define SPLIT_T_NTX 512
#define SPLIT_T_NTY 1
#define SPLIT_VEC_NTX 512
#define SPLIT_VEC_NBX 512

#define CONST 0.75 // for splitting

template <typename TYPE>
__device__
TYPE NextPowTwo (const TYPE p) {
	constexpr int32_t epse_type = getEpse <TYPE> ();
	return scalbn (p, epse_type) - (TYPE)((scalbn (1., epse_type) - 1) * p);
}


template <typename TYPE1, typename TYPE2>
__host__ __device__
int32_t getRho (const int32_t dim, const int32_t overflowMode) {
	constexpr int32_t epse_type1 = getEpse <TYPE1> ();
	constexpr int32_t epse_type2 = getEpse <TYPE2> ();
	if (overflowMode)
		return ceil((epse_type1-(epse_type2-log2(2.*sqrt(dim)))/2)); // overflow-ver
	else
		return ceil((epse_type1-(epse_type2-log2(1.*dim))/2)); // standard-ver
}

// =========================================

template <typename TYPE>
__global__
void cuozblasFindMaxRKernel (
	const int32_t m,
	const int32_t n,
	const TYPE * __restrict__ devInput,
	const int32_t ldi,
	TYPE *devMax
) {
	const int32_t iBx = blockIdx.x;
	const int32_t nTx = SPLIT_N_NTX;//blockDim.x; 
	const int32_t nTy = SPLIT_N_NTY;//blockDim.y;
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t addrx = iBx * nTx + iTx;
	TYPE max, input, tmp;
	__shared__ TYPE shm[nTx*nTy];
	int32_t j;

	if (addrx < m){
		max = 0.;
		for (j = iTy; j < n; j+=nTy) {
			input = devInput[j * ldi + addrx];
			//if (max < fabs(input)) max = fabs(input);
            max += input;
		}
		//shm[nTx * iTy + iTx] = max;	
		__syncthreads ();
		if (iTy == 0) {
            /*
			max = shm[iTx];
			#pragma unroll
			for (j = 1; j < SPLIT_N_NTY; j++) {
				tmp = shm[nTx * j + iTx];
				if (max < fabs(tmp)) max = fabs(tmp);
			}
            */
			devMax[addrx] = max;
		}
	}
}

template <typename TYPE>
__global__
void cuozblasFindMaxCKernel (
	const int32_t m,
	const int32_t n,
	const TYPE * __restrict__ devInput,
	const int32_t ldi,
	TYPE *devMax
) {
	const int32_t iBx = blockIdx.x;
	const int32_t iBy = blockIdx.y;
	const int32_t nTx = SPLIT_T_NTX;//blockDim.x;
	const int32_t nTy = SPLIT_T_NTY;//blockDim.y;
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t addrx = iBx * nTx + iTx;
	const int32_t addry = iBy * nTy + iTy;
	TYPE max, input, tmp;
	__shared__ TYPE shm[nTx];

	if (addry < n){
		max = 0.;
		for (int32_t i = addrx; i < m; i += nTx) {
			input = devInput[addry * ldi + i];
			if (max < fabs(input)) max = fabs(input);
		}
		shm[iTx] = max;
		__syncthreads ();
		#pragma unroll
		for (int32_t i = nTx/2; i > 0; i>>=1) {
			tmp = shm[iTx+i];
			if (iTx < i && shm[iTx] < tmp) shm[iTx] = tmp;
			__syncthreads ();
		}
		if (iTx == 0) devMax[addry] = shm[0];
	} 
}

template <typename TYPE>
__host__
void cuozblasFindMaxDevice (
	const char major,
	const int32_t m, 
	const int32_t n,
	const TYPE * __restrict__ devInput,
	const int32_t ldi, 
	TYPE *devMax
) {
	int32_t ntx, nty, nbx, nby;
	dim3 threads, grid;
	if (major == 'r') {
		ntx = SPLIT_N_NTX;
		nty = SPLIT_N_NTY;
		nbx = ceil (float(m) / ntx);
		nby = 1;
		threads = dim3 (ntx, nty);
		grid = dim3 (nbx, nby);
		cuozblasFindMaxRKernel <<< grid, threads >>> (m, n, devInput, ldi, devMax);
	} else {
		ntx = SPLIT_T_NTX;
		nty = SPLIT_T_NTY;
		nbx = 1;
		nby = ceil (float(n) / nty);
		threads = dim3 (ntx, nty);
		grid = dim3 (nbx, nby);
		cuozblasFindMaxCKernel <<< grid, threads >>>(m, n, devInput, ldi, devMax);
	}
}

// max for n
template <typename TYPE>
__global__
void cuozblasSplitRKernel (
	const int32_t m,
	const int32_t n,
	const int32_t rho,
	const TYPE * __restrict__ devInput,
	const int32_t ldi,
	TYPE *devOutput,
	const int32_t ldo,
	TYPE *devSplit,
	const int32_t lds,
	short *devSpExp,
	TYPE *devMax,
	const int32_t splitShift
) {
	const int32_t iBx = blockIdx.x;
	const int32_t nTx = SPLIT_N_NTX;//blockDim.x;
	const int32_t nTy = SPLIT_N_NTY;//blockDim.y;
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t addrx = iBx * nTx + iTx;
	__shared__ double shm[nTx*nTy];

	if (addrx < m){
		//const TYPE sigma = CONST * scalbn (1., rho) * NextPowTwo <TYPE> (devMax[addrx]) / splitShift;
		TYPE max_ = 0.;
		
        for (int32_t j = iTy; j < n; j += nTy) {
            printf("dev j=%d ldi=%d addrx=%d\n", j, ldi, addrx);
			TYPE input = devInput[j * ldi + addrx];
			//const TYPE split = SUB (ADD (input, sigma), sigma);
			//input = SUB (input, split);
			//devSplit[j * lds + addrx] = split;
			//devOutput[j * ldo + addrx] = input;
			max_ += input;//MAX(max_, fabs(input));
		}
		
		//shm[nTx * iTy + iTx] = max_;	
        __syncthreads ();
		if (iTy == 0) {
            /*
			max_ = shm[iTx];
			#pragma unroll
			for (int32_t j = 1; j < nTy; j++) 
				max_ = MAX(max_, fabs(shm[nTx * j + iTx]));
            */
			devMax[addrx] = max_;
		}
	}
}

template <typename TYPE1, typename TYPE2>
__global__
void cuozblasSplitRKernel (
	const int32_t m,
	const int32_t n,
	const int32_t rho,
	const TYPE1 * __restrict__ devInput,
	const int32_t ldi,
	TYPE1 *devOutput,
	const int32_t ldo,
	TYPE2 *devSplit,
	const int32_t lds,
	short *devSpExp,
	TYPE1 *devMax,
	const int32_t splitShift
) {
}

// max for m
template <typename TYPE>
__global__
void cuozblasSplitCKernel (
	const int32_t m,
	const int32_t n,
	const int32_t rho,
	const TYPE * __restrict__ devInput,
	const int32_t ldi,
	TYPE *devOutput,
	const int32_t ldo,
	TYPE *devSplit,
	const int32_t lds,
	short *devSpExp,
	TYPE *devMax,
	const int32_t splitShift
) {
	const int32_t nTx = SPLIT_T_NTX;//blockDim.x;
	const int32_t iTx = threadIdx.x;
	const int32_t addrx = iTx;

	if (addry < n)
    {
		//const TYPE sigma = CONST * scalbn (1., rho) * NextPowTwo <TYPE> (devMax[addry]) / splitShift;
		TYPE max = 0.;
		__syncthreads ();
        
        #pragma unroll
		for (int32_t i = nTx/2; i > 0; i>>=1) {
			//if (iTx < i) shm[iTx] += shm[iTx+i];
            max++;
			__syncthreads ();
		}
        //if (addrx < n)
        {
            devMax[addrx] = max;
        }
	}
}

template <typename TYPE1, typename TYPE2>
__global__
void cuozblasSplitCKernel (
	const int32_t m,
	const int32_t n,
	const int32_t rho,
	const TYPE1 * __restrict__ devInput,
	const int32_t ldi,
	TYPE1 *devOutput,
	const int32_t ldo,
	TYPE2 *devSplit,
	const int32_t lds,
	short *devSpExp,
	TYPE1 *devMax,
	const int32_t splitShift
) {
    /*
	const int32_t iBx = blockIdx.x;
	const int32_t iBy = blockIdx.y;
	const int32_t nTx = SPLIT_T_NTX;//blockDim.x;
	const int32_t nTy = SPLIT_T_NTY;//blockDim.y;
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t addrx = iBx * nTx + iTx;
	const int32_t addry = iBy * nTy + iTy;
	__shared__ double shm[nTx];

	if (addry < n){
		const short tau = devSpExp[addry] = ceil(log2(fabs(devMax[addry])));
		const TYPE1 sigma = MUL (MUL (CONST, scalbn (1., rho + tau)), splitShift);
		TYPE1 max = 0.;
		for (int32_t i = addrx; i < m; i += nTx) {
			TYPE1 input = devInput[addry * ldi + i];
			const TYPE1 split = SUB (ADD (input, sigma), sigma);
			input = SUB (input, split);
			devSplit[addry * lds + i] = scalbn(split, -tau);
			devOutput[addry * ldo + i] = input;
			max = MAX(max, fabs(input));
		}
		shm[iTx] = max;
		__syncthreads ();
		#pragma unroll
		for (int32_t i = nTx/2; i > 0; i>>=1) {
			if (iTx < i && shm[iTx] < shm[iTx+i]) shm[iTx] = shm[iTx+i];
			__syncthreads ();
		}
		if (iTx == 0) devMax[addry] = shm[0];
	}
    */
}

template <typename TYPE1, typename TYPE2>
__host__
void cuozblasSplitDevice (
	cuozblasHandle_t *oh,
	const char major,
	const int32_t m, 
	const int32_t n,
	const TYPE1 *devInput, // input matrix (devAwrk) 
	const int32_t ldi, // leading dimension of input matrix
	TYPE1 *devOutput, // output matrix (devAwrk)
	const int32_t ldo,
	TYPE2 *devSplit, // split matrices (output): this includes NumSplitMax matrices
	const int32_t lds, // leading dimension of split matrix (# of cols)
	short *devSpExp, // exponent of split matrix
	TYPE1 *devMax,
	int32_t overflowModeFlag,
	const int32_t splitShift
) {
	int32_t ntx, nty, nbx, nby;
	dim3 threads, grid;
	const int32_t dim = (major == 'r') ? n : m;
	const int32_t rho = getRho <TYPE1, TYPE2> (dim, overflowModeFlag);

	if (major == 'r') {
		ntx = SPLIT_N_NTX;
		nty = SPLIT_N_NTY;
		nbx = ceil (float(m) / ntx);
		nby = 1;
		threads = dim3 (ntx, nty);
		grid = dim3 (nbx, nby);
		cuozblasSplitRKernel <<< grid, threads >>> (m, n, rho, devInput, ldi, devOutput, ldo, devSplit, lds, devSpExp, devMax, splitShift);
	} else {
		ntx = SPLIT_T_NTX;
		nty = SPLIT_T_NTY;
		nbx = 1;
		nby = ceil (float(n) / nty);
		threads = dim3 (ntx, nty);
		grid = dim3 (nbx, nby);
		cuozblasSplitCKernel <<< grid, threads >>> (m, n, rho, devInput, ldi, devOutput, ldo, devSplit, lds, devSpExp, devMax, splitShift);
	} 
}

template <typename TYPE1, typename TYPE2>
__host__
int32_t cuozblasSplit (
	cuozblasHandle_t *oh,
	const char major,
	const int32_t m,
	const int32_t n,
	const TYPE1 *devInput,
	const int32_t ldi,
	TYPE1 *devOutput,
	const int32_t ldo,
	TYPE2 *devSplit,
	const int32_t lds,
	short *devSpExp, 
	const int32_t ldse,
	TYPE1 *devMax
) {
	// FindMax^(0)
	if ((major == 'r' && m == 1) || (major == 'c' && n == 1)) {
		int32_t ptrMax = 0;
		blasRiamax (oh->ch, ((major == 'r') ? n : m), devInput, 1, &ptrMax);
		cudaMemcpy (devMax, &devInput[ptrMax-1], sizeof(TYPE1), cudaMemcpyDeviceToDevice);
	} else {
		cuozblasFindMaxDevice (major, m, n, devInput, ldi, devMax);
	}
	int32_t s = 1;
	cuozblasSplitDevice (oh, major, m, n, devInput, ldi, devOutput, ldo, devSplit, lds, devSpExp, devMax, oh->overflowModeFlag, oh->splitShift);
#if 0
	// Split^(0) & FindMax^(1)
	const int32_t maxS = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
	for (s = 1; s < maxS; s++) {
		TYPE1 check = 0.;
		if ((major == 'r' && m == 1) || (major == 'c' && n == 1))
			cudaMemcpy (&check, devMax, sizeof(TYPE1), cudaMemcpyDeviceToHost);
		else
			blasRasum (oh->ch, ((major == 'r') ? m : n), devMax, 1, &check);
		if (check == 0.) return s;
		// Split^(i) & FindMax^(i+1)
		cuozblasSplitDevice (oh, major, m, n, devOutput, ldo, devOutput, ldo, &devSplit[lds*n*s], lds, &devSpExp[ldse*s], devMax, oh->overflowModeFlag, oh->splitShift);
	}
#endif
	if (oh->splitModeFlag > 0)
		fprintf (OUTPUT, "OzBLAS error: infSplit is failed.\n");
	return s;
}
template int32_t cuozblasSplit <double, double> (cuozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const double *devInput, const int32_t ldi, double *devOutput, const int32_t ldo, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax);
template int32_t cuozblasSplit <double, float> (cuozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const double *devInput, const int32_t ldi, double *devOutput, const int32_t ldo, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax);
template int32_t cuozblasSplit <float, float> (cuozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const float *devInput, const int32_t ldi, float *devOutput, const int32_t ldo, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax);
template int32_t cuozblasSplit <float, double> (cuozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const float *devInput, const int32_t ldi, float *devOutput, const int32_t ldo, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax);

