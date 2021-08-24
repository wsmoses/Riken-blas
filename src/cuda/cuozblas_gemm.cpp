#include "cuozblas_common.h"

#define cuozblasMatAddrAlloc(oh, m, n, T, ddev, ld)\
{\
	/*ld = cugetPitchSize (m);*/\
	ld =  ceil((float)m / 128) * 128;\
	ddev = new T[ld * n];\
}

#define cuozblasVecAddrAlloc(oh, n, T, ddev)\
{\
	/*int32_t ld = cugetPitchSize(n);*/\
	int32_t ld =  ceil((float)n / 128) * 128;\
	ddev = new T[ld];\
}

template <typename TYPE1, typename TYPE2>
int32_t cuozblasRgemm (
	cuozblasHandle_t *oh,	
	const char transA, const char transB,
	const int32_t m, const int32_t n, const int32_t k,
	const TYPE1 alpha,
	const TYPE1 *devA, const int32_t lda,
	const TYPE1 *devB, const int32_t ldb,
	const TYPE1 beta,
	TYPE1 *devC, const int32_t ldc
) {
	if (oh->reproModeFlag == 0 && oh->nSplitMax == 1) {
		blasRgemm (oh->ch, transA, transB, m, n, k, alpha, (TYPE1*)devA, lda, (TYPE1*)devB, ldb, beta, devC, ldc);
		return 0;
	}
	cucounterInit (oh);
	double t1, t0 = cutimer();

    /*
	TYPE1 *__attribute__((aligned(8192))) devATmp;
    TYPE1 *__attribute__((aligned(8192))) devBTmp;
    TYPE1 *__attribute__((aligned(8192))) devCTmp;
	TYPE1 *__attribute__((aligned(8192))) devAmax;
    TYPE1 *__attribute__((aligned(8192))) devBmax;
	TYPE2 *__attribute__((aligned(8192))) devASplit;
    TYPE2 *__attribute__((aligned(8192))) devBSplit;
    TYPE2 *__attribute__((aligned(8192))) devCSplit;
    */
	
    TYPE1 *devATmp;
    TYPE1 *devBTmp;
    TYPE1 *devCTmp;
	TYPE1 *devAmax;
    TYPE1 *devBmax;
	TYPE2 *devASplit;
    TYPE2 *devBSplit;
    TYPE2 *devCSplit;
	TYPE2 fone = 1., fzero = 0.;
	short *devASpExp, *devBSpExp;
	int32_t ldas, ldbs, ldcs, ldase, ldbse, ldat, ldbt, ldct;
	int32_t mbk = m;
	int32_t nbk = n;
	int32_t nSplitMaxLoc = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
	int32_t sizeType1 = sizeof (TYPE1);
	int32_t sizeType2 = sizeof (TYPE2);
	int32_t sizeTypeS = sizeof (short);

	// Memory allocation 
	int32_t memorysize = 0;
	TYPE2 **batchAptr    = new TYPE2*[nSplitMaxLoc * nSplitMaxLoc];
	TYPE2 **batchAptrHst    = new TYPE2*[nSplitMaxLoc * nSplitMaxLoc];
	memorysize  += sizeof(TYPE2*) * nSplitMaxLoc * nSplitMaxLoc;
	TYPE2 **batchBptr    = new TYPE2*[nSplitMaxLoc * nSplitMaxLoc];
	TYPE2 **batchBptrHst    = new TYPE2*[nSplitMaxLoc * nSplitMaxLoc];
	memorysize  += sizeof(TYPE2*) * nSplitMaxLoc * nSplitMaxLoc;
	TYPE2 **batchCptr    = new TYPE2*[nSplitMaxLoc * nSplitMaxLoc];
	TYPE2 **batchCptrHst    = new TYPE2*[nSplitMaxLoc * nSplitMaxLoc];
	/*
    int32_t memorysize = 0;
	TYPE2 **batchAptr    = (TYPE2**)(oh->devBatchAddr);
	TYPE2 **batchAptrHst = (TYPE2**)(oh->hstBatchAddr);
	memorysize  += sizeof(TYPE2*) * nSplitMaxLoc * nSplitMaxLoc;
	TYPE2 **batchBptr    = (TYPE2**)(oh->devBatchAddr + memorysize);
	TYPE2 **batchBptrHst = (TYPE2**)(oh->hstBatchAddr + memorysize);
	memorysize  += sizeof(TYPE2*) * nSplitMaxLoc * nSplitMaxLoc;
	TYPE2 **batchCptr    = (TYPE2**)(oh->devBatchAddr + memorysize);
	TYPE2 **batchCptrHst = (TYPE2**)(oh->hstBatchAddr + memorysize);
    */

	int32_t memAddrTmp = oh->memAddr;
	while (mbk > 0 && nbk > 0) { // blocking
		int32_t sizeCn = nbk * nSplitMaxLoc * nSplitMaxLoc;
		if (cucheckTrans (transA) == 0) {
			cuozblasMatAddrAlloc (oh, mbk, k * nSplitMaxLoc, TYPE2, devASplit, ldas); 
            cuozblasMatAddrAlloc (oh, mbk, k,                TYPE1, devATmp,   ldat); 
		} else {
			cuozblasMatAddrAlloc (oh, k, mbk * nSplitMaxLoc, TYPE2, devASplit, ldas); 
			cuozblasMatAddrAlloc (oh, k, mbk,                TYPE1, devATmp,   ldat); 
		}
		if (cucheckTrans (transB) == 0) {
			cuozblasMatAddrAlloc (oh, k, nbk * nSplitMaxLoc, TYPE2, devBSplit, ldbs);
			cuozblasMatAddrAlloc (oh, k, nbk,                TYPE1, devBTmp,   ldbt); 
		} else {
			cuozblasMatAddrAlloc (oh, nbk, k * nSplitMaxLoc, TYPE2, devBSplit, ldbs);
			cuozblasMatAddrAlloc (oh, nbk, k,                TYPE1, devBTmp,   ldbt);
		}
		cuozblasMatAddrAlloc (oh, mbk, sizeCn,           TYPE2, devCSplit, ldcs);
		cuozblasMatAddrAlloc (oh, mbk, nbk,              TYPE1, devCTmp,   ldct);
		// Exp
		cuozblasMatAddrAlloc (oh, mbk, nSplitMaxLoc, short, devASpExp, ldase);
		cuozblasMatAddrAlloc (oh, nbk, nSplitMaxLoc, short, devBSpExp, ldbse);
		// Splitting
		cuozblasVecAddrAlloc (oh, mbk, TYPE1, devAmax);
		cuozblasVecAddrAlloc (oh, nbk, TYPE1, devBmax);
		if (!cumemCheck (oh)) break; // check if work-memory is enough or not
		oh->memAddr = memAddrTmp;
		mbk = ceil (mbk / 2.);
		nbk = ceil (nbk / 2.);
	}
	oh->mbk = mbk;
	oh->nbk = nbk;

	// main part (Split, Comp, Sum)
	int32_t block_count = 0;
	int32_t im = 0;
    //for (int32_t im = 0; im < ceil((float)m/mbk); im++) 
    {
		int32_t mbk_ = (m-mbk*im >= mbk) ? mbk : m-mbk*im;
		// SplitA -----------------------------------
		//t1 = cutimer();
		int32_t nSplitA = 0;
		//if (cucheckTrans (transA) == 0) 
			nSplitA = cuozblasSplit (oh, 'r', mbk_, k, devA+im*mbk, lda, devATmp, ldat, devASplit, ldas, devASpExp, ldase, devAmax);
		//else 
		//	nSplitA = cuozblasSplit (oh, 'c', k, mbk_, devA+im*mbk*lda, lda, devATmp, ldat, devASplit, ldas, devASpExp, ldase, devAmax);
		//oh->t_SplitA += cutimer() - t1;

#if 0
		for (int32_t in = 0; in < ceil((float)n/nbk); in++) {
			int32_t nbk_ = (n-nbk*in >= nbk) ? nbk : n-nbk*in;
			// SplitB -----------------------------------
			t1 = cutimer();
			int32_t nSplitB = 0;
			if (cucheckTrans (transB) == 0) 
				nSplitB = cuozblasSplit (oh, 'c', k, nbk_, devB+in*nbk*ldb, ldb, devBTmp, ldbt, devBSplit, ldbs, devBSpExp, ldbse, devBmax);
			else
				nSplitB = cuozblasSplit (oh, 'r', nbk_, k, devB+in*nbk, ldb, devBTmp, ldbt, devBSplit, ldbs, devBSpExp, ldbse, devBmax);
			oh->t_SplitB += cutimer() - t1;
			// Compute --------------------------------------
			t1 = cutimer();
			double t_sum_local = 0.;
			int32_t ia, ib, ic, ik;
			const int32_t maxlevel = (oh->fastModeFlag) ? MIN (nSplitA-1, nSplitB-1) : (nSplitA-1) + (nSplitB-1);

			// Check num of GEMMs 
			ic = 0;
			for (ia = 0; ia < nSplitA; ia++) {
				for (ib = 0; ib < nSplitB; ib++) {
					if (ia + ib <= maxlevel) {
						oh->n_comp += 2. * mbk_ * nbk_ * k;
					}
				}
			}
			int32_t numB;
			ic = 0;
			if (n == 1 && oh->fastModeFlag == 0) { // DOT & GEMV with fast=0
				for (ia = 0; ia < MIN (maxlevel+1, nSplitA); ia++) {
					numB = MIN (nSplitB, maxlevel+1 - ia);
					batchAptrHst[ic] = devASplit+ldas*((cucheckTrans(transA) == 0) ? k:mbk)*ia;
					batchBptrHst[ic] = devBSplit;
					batchCptrHst[ic] = devCSplit+ldcs*numB*ic; // as nbk=1
					ic++;
				}
			} else { // GEMM, DOT & GEMV with fast=1
				for (ik = 0; ik <= maxlevel; ik++) {
					for (ia = 0; ia < nSplitA; ia++) {
						for (ib = 0; ib < nSplitB; ib++) {
							if (ik == ia + ib) {
								batchAptrHst[ic] = devASplit+ldas*((cucheckTrans(transA) == 0) ? k:mbk)*ia;
								batchBptrHst[ic] = devBSplit+ldbs*((cucheckTrans(transB) == 0) ? nbk:k)*ib;
								batchCptrHst[ic] = devCSplit+ldcs*nbk*ic;
								ic++;
							}
						}
					}
				}
			}
			int32_t nSplitC = ic;
			cudaMemcpy(batchAptr, batchAptrHst, sizeof(TYPE2*) * nSplitC, cudaMemcpyHostToDevice);
			cudaMemcpy(batchBptr, batchBptrHst, sizeof(TYPE2*) * nSplitC, cudaMemcpyHostToDevice);
			cudaMemcpy(batchCptr, batchCptrHst, sizeof(TYPE2*) * nSplitC, cudaMemcpyHostToDevice);
			int32_t n_ = (n == 1 && oh->fastModeFlag == 0) ? numB : nbk_;
			blasRgemmBatch (oh->ch, transA, transB, mbk_, n_, k, fone, (const TYPE2**)batchAptr, ldas,
							(const TYPE2**)batchBptr, ldbs, fzero, (TYPE2**)batchCptrHst, ldcs, 1, nSplitC);
			oh->t_comp += cutimer() - t1;
			oh->t_comp -= t_sum_local;
			oh->t_sum += t_sum_local;

			// Sum -----------------------------------------
			t1 = cutimer();
			int32_t sumorder = (m == 1 && n == 1 && oh->fastModeFlag == 0) ? 2 : 1; // DOT w/o fastmode = 2
			cuozblasGlobalSum (oh, mbk_, nbk_, devASpExp, ldase, nSplitA, devBSpExp, ldbse, nSplitB,
							devCSplit, ldcs*nbk, ldcs, &devC[ldc*(in*nbk)+im*mbk], ldc, alpha, beta, maxlevel, sumorder);
			oh->t_sum += cutimer() - t1;
			block_count++;
			oh->nSplitA += nSplitA;
			oh->nSplitB += nSplitB;
			oh->nSplitC += nSplitC;
		} // EndFor (in)
#endif
	} // EndFor (im)

	//oh->t_total = cutimer() - t0;
	//oh->nSplitA /= (float)block_count;
	//oh->nSplitB /= (float)block_count;
	//oh->nSplitC /= (float)block_count;

	return 0;
}

template int32_t cuozblasRgemm <double, double> (cuozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double *devA, const int32_t lda, const double *devB, const int32_t ldb, const double beta, double *devC, const int32_t ldc);
template int32_t cuozblasRgemm <double, float> (cuozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double *devA, const int32_t lda, const double *devB, const int32_t ldb, const double beta, double *devC, const int32_t ldc);
template int32_t cuozblasRgemm <float, float> (cuozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float *devA, const int32_t lda, const float *devB, const int32_t ldb, const float beta, float *devC, const int32_t ldc); 
template int32_t cuozblasRgemm <float, double> (cuozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float *devA, const int32_t lda, const float *devB, const int32_t ldb, const float beta, float *devC, const int32_t ldc);
