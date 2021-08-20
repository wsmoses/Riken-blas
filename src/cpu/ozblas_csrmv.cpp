#include "ozblas_common.h"

template <typename TYPE1, typename TYPE2>
int32_t ozblasRcsrmv (
	ozblasHandle_t *oh,
	const char tranA, 
	const int32_t m,
	const int32_t n,
	const int32_t nnz,
	const TYPE1 alpha,
	const char *descrA, 
	const TYPE1 *devA,
	const int32_t *devAcolind,
	const int32_t *devArowptr,
	const TYPE1 *devB,
	const TYPE1 beta,
	TYPE1 *devC
) {
	if (oh->reproModeFlag == 0 && oh->nSplitMax == 1) {
		double t0 = timer();
		blasRcsrmv (tranA, m, n, alpha, descrA, devA, devAcolind, devArowptr, devB, beta, devC);
		oh->t_SpMV_SpMM_total += timer() - t0;
		return 0;
	}
	if (tranA == 't' || tranA == 'T') {
		fprintf (OUTPUT, "error: transposed mode is not implemented.\n");
		exit (1);
	}
	counterInit (oh);
	double t1, t0 = timer();
	short *devASpExp, *devBSpExp;
	TYPE2 fone = 1., fzero = 0.;
	TYPE2 *devASplit, *devBSplit, *devCSplit;
	TYPE1 *devATmp, *devBTmp, *devCTmp;
	TYPE1 *devAmax_, *devBmax_;
//	TYPE2 *devAmax, *devATmpD1, *devATmpD2, *devATmpD3;
	TYPE2 *devBmax, *devBTmpD1, *devBTmpD2, *devBTmpD3;
	#if defined (FLOAT128) 
	double *devCTmp1, *devCTmp2, *devCTmp3;
	int32_t sizeTypeT = sizeof (double);
	#else
	TYPE2 *devCTmp1, *devCTmp2, *devCTmp3;
	int32_t sizeTypeT = sizeof (TYPE2);
	#endif
	int32_t ldas, ldbs, ldcs, ldase;
	int32_t sizeType1 = sizeof (TYPE1);
	int32_t sizeType2 = sizeof (TYPE2);
	int32_t sizeTypeS = sizeof (short);
	int32_t nSplitMaxLoc = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;

	if (oh->memMaskSplitA != 0) oh->memAddr = 0;
	// --- here is preserved ---
	ozblasMatAddrAlloc (oh, nnz, nSplitMaxLoc, sizeType2, (void**)&devASplit, ldas);
	ozblasVecAddrAlloc (oh, nnz, sizeType1, (void**)&devATmp);
	ozblasVecAddrAlloc (oh, m,   sizeType1, (void**)&devAmax_);
	ozblasMatAddrAlloc (oh, m, nSplitMaxLoc, sizeTypeS, (void**)&devASpExp, ldase);
	//if (oh->splitModeFlag == 3) 
	//		ozblasVecAddrAlloc (oh, m, sizeType2, (void**)&devAmax);
	// --- here is preserved ---
	if (oh->memMaskSplitA != 0) oh->memMaskSplitA = oh->memAddr;

	ozblasMatAddrAlloc (oh, n, nSplitMaxLoc, sizeType2, (void**)&devBSplit, ldbs);
	ozblasMatAddrAlloc (oh, m, nSplitMaxLoc * nSplitMaxLoc, sizeType2, (void**)&devCSplit, ldcs);
	ozblasVecAddrAlloc (oh, n, sizeType1, (void**)&devBTmp);
	ozblasVecAddrAlloc (oh, m, sizeType1, (void**)&devCTmp);
	if (oh->sumModeFlag == 3) {
		ozblasVecAddrAlloc (oh, m, sizeTypeT, (void**)&devCTmp1);
		ozblasVecAddrAlloc (oh, m, sizeTypeT, (void**)&devCTmp2);
		ozblasVecAddrAlloc (oh, m, sizeTypeT, (void**)&devCTmp3);
	}
	// Exp
	ozblasVecAddrAlloc (oh, nSplitMaxLoc, sizeTypeS, (void**)&devBSpExp);
	// Splitting
	ozblasVecAddrAlloc (oh, 1, sizeType1, (void**)&devBmax_);
	// above must be allocated even if splitModeFlag is 3 as they may be used if Split3 is not used
	if (oh->splitModeFlag == 3) {
		// Currently, split3 is only for B
		//ozblasVecAddrAlloc (oh, nnz, sizeType2, (void**)&devATmpD1); 
		//ozblasVecAddrAlloc (oh, nnz, sizeType2, (void**)&devATmpD2); 
		//ozblasVecAddrAlloc (oh, nnz, sizeType2, (void**)&devATmpD3); 
		ozblasVecAddrAlloc (oh, 1, sizeType2, (void**)&devBmax);
		ozblasVecAddrAlloc (oh, n, sizeType2, (void**)&devBTmpD1);
		ozblasVecAddrAlloc (oh, n, sizeType2, (void**)&devBTmpD2);
		ozblasVecAddrAlloc (oh, n, sizeType2, (void**)&devBTmpD3);
	}
	
	if (memCheck (oh)) {
		fprintf (OUTPUT, "OzBLAS error: memory shortage.\n");
		exit (1);
	}

	// Split of A -----------------------------------
	t1 = timer();
	if (oh->splitModeFlag == 3) {
		fprintf (OUTPUT, "OzBLAS warning: split3 for sparse matrix is not implemented.\n");
		exit (1);
	}
	int32_t nSplitA;
	if (oh->memMaskSplitA == 0) {
		nSplitA = ozblasSplitSparse (oh, 'r', m, devA, devArowptr, devATmp, devASplit, ldas, devASpExp, ldase, devAmax_);
	} else {
		devASplit = (TYPE2*)devA;
		nSplitA = oh->nSplitA_;
	}
	oh->t_SplitA += timer() - t1;

	// Split of B -----------------------------------
	t1 = timer();
	int32_t split3FlagB = (oh->splitModeFlag == 3) ? rangeCheck <TYPE1, TYPE2> (n, 1, devB, n) : 0; // on (if 1)
	int32_t nSplitB;
	if (split3FlagB) 
		nSplitB = ozblasSplit3 (oh, 'c', n, 1, devB, n, devBSplit, ldbs, devBSpExp, 1, devBmax,
								devBTmpD1, n, devBTmpD2, n, devBTmpD3, n, devBTmpD1, n);
	else
		nSplitB = ozblasSplit (oh, 'c', n, 1, devB, n, devBTmp, n, devBSplit, ldbs, devBSpExp, 1, devBmax_);
	oh->t_SplitB += timer() - t1;

	// Compute --------------------------------------
	t1 = timer();
	int32_t ia, ib, ic, ik;
	const int32_t maxlevel = (nSplitA-1) + (nSplitB-1);
	TYPE2 *ptrB = devBSplit;
	TYPE2 *ptrC = devCSplit;
	ic = 0;
	for (ia = 0; ia < MIN (maxlevel+1, nSplitA); ia++) {
		const int32_t numB = MIN (nSplitB, maxlevel+1 - ia);
		TYPE2 *ptrA = devASplit+ldas*ia;
		blasRcsrmm (tranA, m, numB, n, fone, descrA, ptrA, devAcolind, devArowptr, ptrB, ldbs, fzero, ptrC, ldcs);
		ptrC += ldcs*numB;
		ic += numB;
	}
	const int32_t nSplitC = ic;
	oh->nSplitA += nSplitA;
	oh->nSplitB += nSplitB;
	oh->nSplitC += nSplitC;
	oh->t_comp += timer() - t1;
	
	// Sum -----------------------------------------
	t1 = timer();
	ic = 0;
	if (oh->sumModeFlag == 3) {
		for (ik = 0; ik <= maxlevel; ik++) {
			for (ia = 0; ia < nSplitA; ia++) {
				for (ib = 0; ib < nSplitB; ib++) {
					if (ik == ia + ib) {
						int32_t it = nSplitB * ia + ib; // unlike GEMV, here is transposed
						if (ozblasLocalFsum3 (m, 1, &devASpExp[ldase*ia], &devBSpExp[ib], devCSplit+ldcs*it, ldcs,
											devCTmp, m, devCTmp1, m, devCTmp2, m, devCTmp3, m, (ic==nSplitC-1)?-1:ic)) {
							fprintf (OUTPUT, "OzBLAS error: Sum3 is failed.\n");
							exit (1);
						}
						ic++;
					} // EndIf (ik)
				} // EndFor (ib)
			} // EndFor (ia)
		} // EndFor (ik)
		ozblasAxpby (m, 1, devCTmp, 1, devC, 1, alpha, beta);
	} else { // sumMode < 3
		if (ozblasGlobalSum (oh, m, 1, devASpExp, ldase, nSplitA,
							devBSpExp, 1, nSplitB, devCSplit, ldcs, ldcs, devC, 1, alpha, beta, maxlevel, 1)) { 
			fprintf (OUTPUT, "OzBLAS error: sum is failed\n");
			exit (1);
		}
	}
	oh->t_sum += timer() - t1;
	oh->t_total = timer() - t0;

	// for CG, time
	// =================================
	oh->t_SplitMat_total += oh->t_SplitA;
	oh->t_SplitVec_total += oh->t_SplitB;
	oh->t_Sum_total += oh->t_sum;
	oh->t_AXPY_SCAL_total += 0.;
	oh->t_DOT_NRM2_total += 0.;
	oh->t_SpMV_SpMM_total += oh->t_comp;
	// =================================

	return 0;
}
#if defined (FLOAT128)
template int32_t ozblasRcsrmv <__float128, double> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const __float128 alpha, const char *descrA, const __float128 *devA, const int32_t *devAcolind, const int32_t *devArowptr, const __float128 *devB, const __float128 beta, __float128 *devC);
template int32_t ozblasRcsrmv <__float128, float> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const __float128 alpha, const char *descrA, const __float128 *devA, const int32_t *devAcolind, const int32_t *devArowptr, const __float128 *devB, const __float128 beta, __float128 *devC);
#endif
template int32_t ozblasRcsrmv <double, double> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const double alpha, const char *descrA, const double *devA, const int32_t *devAcolind, const int32_t *devArowptr, const double *devB, const double beta, double *devC);
template int32_t ozblasRcsrmv <double, float> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const double alpha, const char *descrA, const double *devA, const int32_t *devAcolind, const int32_t *devArowptr, const double *devB, const double beta, double *devC);
template int32_t ozblasRcsrmv <float, float> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const float alpha, const char *descrA, const float *devA, const int32_t *devAcolind, const int32_t *devArowptr, const float *devB, const float beta, float *devC);
template int32_t ozblasRcsrmv <float, double> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const float alpha, const char *descrA, const float *devA, const int32_t *devAcolind, const int32_t *devArowptr, const float *devB, const float beta, float *devC);


// splitting only (for CG solvers)
template <typename TYPE1, typename TYPE2>
TYPE2 *ozblasRcsrmvSplitA (
	ozblasHandle_t *oh,
	const char tranA, 
	const int32_t m,
	const int32_t n,
	const int32_t nnz,
	const char *descrA, 
	const TYPE1 *devA,
	const int32_t *devArowptr
) {
	if (oh->reproModeFlag == 0 && oh->nSplitMax == 1) {
		return (TYPE2*)devA;
	}
	if (tranA == 't' || tranA == 'T') {
		fprintf (OUTPUT, "OzBLAS error: transposed mode is not implemented.\n");
		exit (1);
	}
	counterInit (oh);
	short *devASpExp;
	TYPE1 *devAmax, *devATmp;
	TYPE2 *devASplit;
	int32_t sizeType1 = sizeof (TYPE1);
	int32_t sizeType2 = sizeof (TYPE2);
	int32_t ldas, ldase;
	int32_t nSplitMaxLoc = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
	// --- here is preserved ---
	ozblasMatAddrAlloc (oh, nnz, nSplitMaxLoc, sizeType2, (void**)&devASplit, ldas);
	ozblasVecAddrAlloc (oh, nnz, sizeType1, (void**)&devATmp);
	ozblasVecAddrAlloc (oh, m, sizeType1, (void**)&devAmax);
	ozblasMatAddrAlloc (oh, m, nSplitMaxLoc, sizeof(short), (void**)&devASpExp, ldase);
	// --- here is preserved ---
	if (memCheck (oh)) {
		fprintf (OUTPUT, "OzBLAS error: memory shortage.\n");
		exit (1);
	}
	oh->memMaskSplitA = oh->memAddr;

	double t1 = timer();
	// Split of A -----------------------------------
	if (oh->splitModeFlag == 3) {
		fprintf (OUTPUT, "OzBLAS error: split3 for sparse matrix is not implemented.\n");
		exit (1);
	}
	int32_t nSplitA = ozblasSplitSparse (oh, 'r', m, devA, devArowptr, devATmp, devASplit, ldas, devASpExp, ldase, devAmax);
	// shiftSize-tuning
	// ------------------------------------------------------------------------
	if (oh->nSplitMax == 0) { // tuning is possible only when full-splitting (d=0)
		printf ("## CSRMV: << shift-size tuning >> num.split = %d -> ", nSplitA);
		oh->splitShift = 1; 
		int32_t nSplitAOld = oh->nSplitA;
		do {
			nSplitAOld = nSplitA;
			oh->splitShift *= 2;
			// try with new shift-size
			nSplitA = nSplitMaxLoc;
			nSplitA = ozblasSplitSparse (oh, 'r', m, devA, devArowptr, devATmp, devASplit, ldas, devASpExp, ldase, devAmax);
		// if numSplit increased, stop
		} while (nSplitAOld == nSplitA);

		// do again with the optimal shift-size
		nSplitA = nSplitMaxLoc;
		nSplitA = ozblasSplitSparse (oh, 'r', m, devA, devArowptr, devATmp, devASplit, ldas, devASpExp, ldase, devAmax);
		printf ("%d (with splitShift = %d)\n", nSplitA, oh->splitShift);
	}
	// ------------------------------------------------------------------------
	oh->nSplitA_ = nSplitA;
	oh->nSplitA = nSplitA;
	oh->t_SplitA += timer() - t1;

	// for CG, time
	// =================================
	oh->t_SplitMat_total += oh->t_SplitA;
	oh->t_SplitVec_total += 0.;
	oh->t_Sum_total += 0.;
	oh->t_AXPY_SCAL_total += 0.;
	oh->t_DOT_NRM2_total += 0.;
	oh->t_SpMV_SpMM_total += 0.;
	// =================================

	return devASplit;
}
#if defined (FLOAT128)
template double *ozblasRcsrmvSplitA <__float128, double> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const char *descrA, const __float128 *devA, const int32_t *devArowptr);
template float *ozblasRcsrmvSplitA <__float128, float> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const char *descrA, const __float128 *devA, const int32_t *devArowptr);
#endif
template double *ozblasRcsrmvSplitA <double, double> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const char *descrA, const double *devA, const int32_t *devArowptr);
template float *ozblasRcsrmvSplitA <double, float> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const char *descrA, const double *devA, const int32_t *devArowptr);
template float *ozblasRcsrmvSplitA <float, float> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const char *descrA, const float *devA, const int32_t *devArowptr);
template double *ozblasRcsrmvSplitA <float, double> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const char *descrA, const float *devA, const int32_t *devArowptr);
