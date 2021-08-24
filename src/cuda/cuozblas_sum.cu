#include "cuozblas_common.h"

// =========================================
// Correctly-rounded summation
// with NearSum
// from "ACCURATE FLOATING-POINT SUMMATION" by S.M.RUMP, T.OGITA, S.OISHI (2005)
// http://www.ti3.tu-harburg.de/paper/rump/RuOgOi06.pdf
// =========================================

template <typename TYPE>
__device__ __forceinline__
void cuozblasTransform (
	const int32_t n,
	TYPE *vec,
	const int32_t ld,
	TYPE rho,
	TYPE &tau1,
	TYPE &tau2
) {
	constexpr int32_t emin = getEmin <TYPE> ();
	constexpr int32_t epse = getEpse <TYPE> ();

	int32_t i, m;
	TYPE tmp, mu, sigma, t, tau;

	mu = fabs(vec[0]);
	for (i = 1; i < n; i++)
		mu = MAX (mu, fabs(vec[i*ld])); 
	if ((n == 0) || (mu == 0.)) {
		tau1 = rho;
		tau2 = 0.;
		return;
	}

	m = ceil (log2((double)(n+2)));
	sigma = scalbn (1., m+ceil(log2(mu)));
	t = rho;
	while (1) {
		// ExtractVector
		tau = 0.;
		for (i = 0; i < n; i++) {
			tmp = SUB (ADD (sigma, vec[i*ld]), sigma);
			vec[i*ld] -= tmp;  // <- output
			tau += tmp;
		}
		// here, tau1 = t1
		tau1 = ADD (t, tau);
		if ((sigma <= scalbn (1., emin)) || (fabs(tau1) >= scalbn (1., 2*m+1-epse)*sigma)) {
			//FastTwoSum (t, tau, &tau1, &tau2);
			//tau1 = t + tau
			tau2 = SUB (tau, SUB (tau1, t));
			return;
		}
		sigma = scalbn (1., m-53) * sigma;
		t = tau1;
	} 
}

template <typename TYPE>
__device__ __forceinline__
void cuozblasTransformK (
	const int32_t n,
	TYPE *vec,
	const int32_t ld,
	TYPE rho,
	TYPE &res,
	TYPE &r
) { 
	TYPE tmp = 0., tau1, tau2;
	cuozblasTransform (n, vec, ld, rho, tau1, tau2);
	for (int32_t i = 0; i < n; i++)
		tmp += vec[i*ld];
	res = ADD (tau1, ADD (tau2, tmp));
	r = SUB (tau2, SUB (res, tau1));
}

template <typename TYPE>
__device__ __forceinline__
TYPE getSign (TYPE v) {
	return (v < 0) ? -1.:1.;
}

template <typename TYPE>
__device__ 
TYPE cuozblasNearsum (
	const int32_t n,
	TYPE *vec,
	const int32_t ld
) {
	constexpr int32_t epse = getEpse <TYPE> ();
	TYPE tmp, res, res2, r, r2, mu, delta, delta2;
	TYPE eps = scalbn (1., -epse);

	cuozblasTransformK (n, vec, ld, (TYPE)0., res, r);
	cuozblasTransformK (n, vec, ld, r, delta, r2);
	if (delta == 0) 
		return res;
	res2 = ADD (res, MUL (getSign (delta), MUL (eps, fabs(res))));
	if (res2 == res) {
		mu = MUL (getSign (delta), MUL (eps, fabs(res)));
		res2 = ADD (res, MUL (2., MUL (getSign (delta), MUL (eps, fabs(res)))));
	} else {
		mu = SUB (res2, res) / 2.;
	}
	if (fabs(delta) < fabs(mu)) 
		return res;
	if (fabs(delta) > fabs(mu)) 
		return res2;
	cuozblasTransformK (n, vec, ld, r2, delta2, tmp);
	if (delta2 == 0) 
		return res + mu;
	if (getSign (delta2) == getSign (mu))
		return res2;
	return res;
}

template <typename TYPE1, typename TYPE2>
__global__
void cuozblasGlobalNearsumKernel (
	const int32_t m,
	const int32_t n,
	const short *devASpExp,
	const int32_t ldase,
	const int32_t nSplitA,
	const short *devBSpExp,
	const int32_t ldbse,
	const int32_t nSplitB,
	TYPE2 *devCsplit,
	const int32_t llsc,
	const int32_t ldsc,
	TYPE1 *devC,
	const int32_t ldc,
	const TYPE1 alpha,
	const TYPE1 beta,
	const int32_t maxlevel,
	const int32_t sumOrder, 
	int32_t *check
) {
	// dummy (not called)
}

template <typename TYPE>
__global__
void cuozblasGlobalNearsumKernel (
	const int32_t m,
	const int32_t n,
	const short *devASpExp,
	const int32_t ldase,
	const int32_t nSplitA,
	const short *devBSpExp,
	const int32_t ldbse,
	const int32_t nSplitB,
	TYPE *devCsplit,
	const int32_t llsc,
	const int32_t ldsc,
	TYPE *devC,
	const int32_t ldc,
	const TYPE alpha,
	const TYPE beta,
	const int32_t maxlevel,
	const int32_t sumOrder, 
	int32_t *check
) {
	const int32_t iBx = blockIdx.x;
	const int32_t iBy = blockIdx.y;
	const int32_t nTx = blockDim.x;
	const int32_t nTy = blockDim.y;
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t addrx = iBx * nTx + iTx;
	const int32_t addry = iBy * nTy + iTy;

	if (addrx < m && addry < n) {
		TYPE t = 0.;
		int32_t ic = 0;
		for (int32_t ik = 0; ik <= maxlevel; ik++) {
			for (int32_t ia = 0; ia < nSplitA; ia++) {
				for (int32_t ib = 0; ib < nSplitB; ib++) {
					if (ik == ia + ib) {
						int32_t it = (sumOrder == 1) ? ic : (nSplitA * ib + ia);
						TYPE c = devCsplit[llsc * it + addry * ldsc + addrx];
						devCsplit[llsc * it + addry * ldsc + addrx] = c;
						ic++;
					}
				}
			}
		}
		t = cuozblasNearsum (ic, &devCsplit[addry * ldsc + addrx], llsc);
		devC[addry * ldc + addrx] = alpha * t + beta * devC[addry * ldc + addrx];
	}
}

// =========================================
// Floating-point summation (FSum)
// =========================================

template <typename TYPE>
__global__
void cuozblasGlobalFsumKernel (
	const int32_t m,
	const int32_t n,
	const short *devASpExp,
	const int32_t ldase,
	const int32_t nSplitA,
	const short *devBSpExp,
	const int32_t ldbse,
	const int32_t nSplitB,
	const TYPE *devCsplit,
	const int32_t llsc,
	const int32_t ldsc,
	TYPE *devC,
	const int32_t ldc,
	const TYPE alpha,
	const TYPE beta,
	const int32_t maxlevel,
	const int32_t sumOrder, 
	int32_t *check
) {
	const int32_t iBx = blockIdx.x;
	const int32_t iBy = blockIdx.y;
	const int32_t nTx = blockDim.x;
	const int32_t nTy = blockDim.y;
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t addrx = iBx * nTx + iTx;
	const int32_t addry = iBy * nTy + iTy;

	if (addrx < m && addry < n) {
		TYPE t = 0.;
		int32_t ic = 0;
		for (int32_t ik = 0; ik <= maxlevel; ik++) {
			for (int32_t ia = 0; ia < nSplitA; ia++) {
				for (int32_t ib = 0; ib < nSplitB; ib++) {
					if (ik == ia + ib) {
						int32_t it = (sumOrder == 1) ? ic : (nSplitA * ib + ia);
						TYPE c = devCsplit[llsc * it + addry * ldsc + addrx];
						t += c;
						ic++;
					}
				}
			}
		}
		devC[addry * ldc + addrx] = alpha * t + beta * devC[addry * ldc + addrx];
	}
}

template <typename TYPE1, typename TYPE2>
__global__
void cuozblasGlobalFsumKernel (
	const int32_t m,
	const int32_t n,
	const short *devASpExp,
	const int32_t ldase,
	const int32_t nSplitA,
	const short *devBSpExp,
	const int32_t ldbse,
	const int32_t nSplitB,
	const TYPE2 *devCsplit,
	const int32_t llsc,
	const int32_t ldsc,
	TYPE1 *devC,
	const int32_t ldc,
	const TYPE1 alpha,
	const TYPE1 beta,
	const int32_t maxlevel,
	const int32_t sumOrder, 
	int32_t *check
) {
	const int32_t iBx = blockIdx.x;
	const int32_t iBy = blockIdx.y;
	const int32_t nTx = blockDim.x;
	const int32_t nTy = blockDim.y;
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t addrx = iBx * nTx + iTx;
	const int32_t addry = iBy * nTy + iTy;

	if (addrx < m && addry < n) {
		TYPE1 t = 0.;
		int32_t ic = 0;
		for (int32_t ik = 0; ik <= maxlevel; ik++) {
			for (int32_t ia = 0; ia < nSplitA; ia++) {
				short seA = devASpExp[ldase*ia+addrx];
				for (int32_t ib = 0; ib < nSplitB; ib++) {
					if (ik == ia + ib) {
						int32_t it = (sumOrder == 1) ? ic : (nSplitA * ib + ia);
						TYPE1 c = (TYPE1)devCsplit[llsc * it + addry * ldsc + addrx];
						short seB = devBSpExp[ldbse*ib+addry];
						t += scalbn (c, seA+seB);
						ic++;
					}
				}
			}
		}
		devC[addry * ldc + addrx] = alpha * t + beta * devC[addry * ldc + addrx];
	}
}

template <typename TYPE1, typename TYPE2>
__host__
int32_t cuozblasGlobalSum (
	cuozblasHandle_t *oh,
	const int32_t m,
	const int32_t n,
	const short *devASpExp,
	const int32_t ldase,
	const int32_t nSplitA,
	const short *devBSpExp,
	const int32_t ldbse,
	const int32_t nSplitB,
	TYPE2 *devCsplit, // [llsc * nSplitA * nSplitB]
	const int32_t llsc, // = ldsc * n
	const int32_t ldsc,
	TYPE1 *devC,
	const int32_t ldc,
	const TYPE1 alpha,
	const TYPE1 beta,
	const int32_t maxlevel,
	const int32_t sumOrder
) {
	dim3 threads, grid;
	int32_t ntx, nty, nbx, nby;
	ntx = 512;
	nty = 1;
	nbx = ceil (float(m) / ntx);
	nby = ceil (float(n) / nty);
	threads = dim3 (ntx, nty);
	grid = dim3 (nbx, nby);
	
	int32_t check = 0;
	if (oh->sumModeFlag == 1) { // Nearsum
		//if (typeid(TYPE1) != typeid(TYPE2)) {
		//	fprintf (OUTPUT, "OzBLAS error: Nearsum is not supported when TYPE1 != TYPE2.\n");
		//	exit (1);
		//} else
        {
			cuozblasGlobalNearsumKernel <<< grid, threads >>> (m, n, devASpExp, ldase, nSplitA, devBSpExp, ldbse, nSplitB,
							  devCsplit, llsc, ldsc, devC, ldc, alpha, beta, maxlevel, sumOrder, &check);
		}
	} else { // Fsum
		cuozblasGlobalFsumKernel <<< grid, threads >>> (m, n, devASpExp, ldase, nSplitA, devBSpExp, ldbse, nSplitB,
							  devCsplit, llsc, ldsc, devC, ldc, alpha, beta, maxlevel, sumOrder, &check);
	}
	return check;
}
template int32_t cuozblasGlobalSum <double, double> (cuozblasHandle_t *oh, const int32_t m, const int32_t n, const short *devASpExp, const int32_t ldase, const int32_t nSplitA, const short *devBSpExp, const int32_t ldbse, const int32_t nSplitB, double *devCsplit, const int32_t llsc, const int32_t ldsc, double *devC, const int32_t ldc, const double alpha, const double beta, const int32_t maxlevel, const int32_t sumOrder);
template int32_t cuozblasGlobalSum <double, float> (cuozblasHandle_t *oh, const int32_t m, const int32_t n, const short *devASpExp, const int32_t ldase, const int32_t nSplitA, const short *devBSpExp, const int32_t ldbse, const int32_t nSplitB, float *devCsplit, const int32_t llsc, const int32_t ldsc, double *devC, const int32_t ldc, const double alpha, const double beta, const int32_t maxlevel, const int32_t sumOrder);
template int32_t cuozblasGlobalSum <float, float> (cuozblasHandle_t *oh, const int32_t m, const int32_t n, const short *devASpExp, const int32_t ldase, const int32_t nSplitA, const short *devBSpExp, const int32_t ldbse, const int32_t nSplitB, float *devCsplit, const int32_t llsc, const int32_t ldsc, float *devC, const int32_t ldc, const float alpha, const float beta, const int32_t maxlevel, const int32_t sumOrder);
template int32_t cuozblasGlobalSum <float, double> (cuozblasHandle_t *oh, const int32_t m, const int32_t n, const short *devASpExp, const int32_t ldase, const int32_t nSplitA, const short *devBSpExp, const int32_t ldbse, const int32_t nSplitB, double *devCsplit, const int32_t llsc, const int32_t ldsc, float *devC, const int32_t ldc, const float alpha, const float beta, const int32_t maxlevel, const int32_t sumOrder);

