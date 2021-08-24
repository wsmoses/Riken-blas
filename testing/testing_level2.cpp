#define ROUTINE_FLOPS(m,n,k)	((double)2.*m*n+3.*m)
#define ROUTINE_BYTES(m,n,k,s)	(((double)2.*m+n+m*n)*s) 
#include "testing_common.h"
#include "testing_common.cpp"

int32_t
main (int32_t argc, char **argv)
{
	int32_t i;
	int32_t lda_hst;
	int32_t cda_dev, cda_hst;
	int32_t rda_dev, rda_hst;
	int32_t vlx_dev, vlx_hst;
	int32_t vly_dev, vly_hst;
	double t0, t1, sec;

	#if defined (MPLAPACK)
    mpfr_set_default_prec(MPFR_PREC);
	mpreal::set_default_prec(MPFR_PREC);
	#endif

// testing setup ------------------------------
	testingHandle_t th;
	#if defined (CUOZBLAS) || defined (OZBLAS)
	ozblasHandle_t ha;
	ozblasCreate (&ha, WORK_MEM_SIZE); 
	testingCreate (argc, argv, &th, &ha);
	#elif defined (CUBLAS)
	cublasHandle_t ha;
	cublasCreate (&ha); 
	cublasSetPointerMode(ha, CUBLAS_POINTER_MODE_HOST);
	testingCreate (argc, argv, &th);
	#else
	testingCreate (argc, argv, &th);
	#endif
	print_info1 (&th);
// --------------------------------------------

// memory setup -------------------------------
	dim_hst_setup (&th);
	rda_hst = th.dim_stop;
	cda_hst = th.dim_stop;
	vlx_hst = th.dim_stop;
	vly_hst = th.dim_stop;
	if (th.tranA == 'N' || th.tranA == 'n') {
		if (th.dim_n_const > 0) vlx_hst = th.dim_n_const;
		if (th.dim_m_const > 0) vly_hst = th.dim_m_const;
	} else {
		if (th.dim_m_const > 0) vlx_hst = th.dim_m_const;
		if (th.dim_n_const > 0) vly_hst = th.dim_n_const;
	}
	if (th.dim_m_const > 0) rda_hst = th.dim_m_const;
	if (th.dim_n_const > 0) cda_hst = th.dim_n_const;
	lda_hst = rda_hst;
	// malloc host memory
	FP_TYPE alpha, beta;
	FP_TYPE *hst_A = new FP_TYPE[lda_hst * cda_hst];
	FP_TYPE *hst_X = new FP_TYPE[vlx_hst * th.incx];
	FP_TYPE *hst_Y = new FP_TYPE[vly_hst * th.incy];
	FP_TYPE *hst_Y_t = new FP_TYPE[vly_hst * th.incy];
	// initialize (1:val, 2:phi, 3:erange)
	mublasInitMat (&th, 1, 1, 1, &alpha, 1., 1, 0);
	mublasInitMat (&th, 1, 1, 1, &beta, 0., 1, 0);
	if (th.trunc != 0)  // for reduced-precision performance evaluation
		printf ("### !!! Truncated inputs !!!\n");
	mublasInitMat (&th, rda_hst, cda_hst, lda_hst, hst_A, th.phi, 3, th.trunc);
	mublasInitMat (&th, vlx_hst * th.incx, 1, 0, hst_X, th.phi, 3, th.trunc);
	mublasInitMat (&th, vly_hst * th.incy, 1, 0, hst_Y, 0., 1, 0);

// --------------------------------------------
	print_info2 (&th);

// evaluation ---------------------------------
	dim_dev_setup (&th);
	while (1) {
		if ((th.dim_m_const == 0 && th.dim_m_dev > th.dim_stop) ||
		    (th.dim_n_const == 0 && th.dim_n_dev > th.dim_stop)) break;
		// dim setup
		if (th.tranA == 'N' || th.tranA == 'n') {
			vlx_dev = th.dim_n_dev;
			vly_dev = th.dim_m_dev;
		} else {
			vlx_dev = th.dim_m_dev;
			vly_dev = th.dim_n_dev;
		}
		rda_dev = th.dim_m_dev;
		cda_dev = th.dim_n_dev;

		FP_TYPE *dev_A = hst_A;
		FP_TYPE *dev_X = hst_X;
		FP_TYPE *dev_Y = hst_Y;
		int32_t lda_dev = lda_hst;
		#if defined (CUDA)
		int32_t sizeType = sizeof (FP_TYPE);
		// malloc device memory
		size_t pitch;
		cudaMallocPitch ((void **) &dev_A, &pitch, sizeType * rda_dev, cda_dev);
		lda_dev = pitch/sizeType;
		cudaMalloc ((void **) &dev_X, sizeType * vlx_dev * th.incx);
		cudaMalloc ((void **) &dev_Y, sizeType * vly_dev * th.incy);
		// memcpy from hst to device
		//cublasSetMatrix (rda_dev, cda_dev, sizeType, hst_A, lda_hst, dev_A, lda_dev);
		//cublasSetVector (vlx_dev * th.incx, sizeType, hst_X, 1, dev_X, 1);
		// ---------------------------------------------
		#endif

		printf ("%d\t%d\t--", th.dim_m_dev, th.dim_n_dev);
		get_routine_theoretial_performance (&th);

		// execution ---------------------------------
		if (th.mode == 'p') {
			for (i = 0; i < WLOOP; i++) // warm up
				trgRgemv (ha, th.tranA, rda_dev, cda_dev, alpha, dev_A, lda_dev, dev_X, 1, beta, dev_Y, 1);
			t0 = gettime ();
			for (i = 0; i < NLOOP; i++) 
				trgRgemv (ha, th.tranA, rda_dev, cda_dev, alpha, dev_A, lda_dev, dev_X, 1, beta, dev_Y, 1);
			t1 = gettime ();
			sec = (t1 - t0) / NLOOP;
			printf ("\t%1.3e\t%1.3e\t%1.3e\t%d", sec, (th.routine_flops / sec) * 1.0e-9, (th.routine_bytes / sec) * 1.0e-9, NLOOP);
		}
		if (th.mode == 'c') {
			trgRgemv (ha, th.tranA, rda_dev, cda_dev, alpha, dev_A, lda_dev, dev_X, 1, beta, dev_Y, 1);
			#if defined (CUDA)
			//cublasGetVector (vly_dev * th.incy, sizeType, dev_Y, 1, hst_Y_t, 1);
			#else
			mublasCopyMat (vly_dev * th.incy, 1, dev_Y, 0, hst_Y_t, 0);
			#endif
		}
		// -------------------------------------------

		#if defined (CUDA)
		cudaFree (dev_A);
		cudaFree (dev_X);
		cudaFree (dev_Y);
		#endif
		if (th.mode == 'c') {
			mpreal alpha_r = 0;
			mpreal beta_r = 0;
			mpreal *hst_A_r = new mpreal[rda_dev * cda_dev];
			mpreal *hst_X_r = new mpreal[vlx_dev * th.incx];
			mpreal *hst_Y_r = new mpreal[vly_dev * th.incy];
			mublasConvMat (1, 1, &alpha, 0, &alpha_r, 0);
			mublasConvMat (1, 1, &beta, 0, &beta_r, 0);
			mublasConvMat (rda_dev, cda_dev, hst_A, lda_hst, hst_A_r, rda_dev);
			mublasConvMat (vlx_dev * th.incx, 1, hst_X, 0, hst_X_r, 0);
			mublasConvMat (vly_dev * th.incy, 1, hst_Y, 0, hst_Y_r, 0);
			refRgemv (th.tranA,rda_dev,cda_dev,alpha_r,hst_A_r,rda_dev,hst_X_r,1,beta_r,hst_Y_r,1);
			mublasCheckMatrix (vly_dev * th.incy, 1, hst_Y_t, vly_dev * th.incy, hst_Y_r, vly_dev * th.incy);
			delete[]hst_A_r;
			delete[]hst_X_r;
			delete[]hst_Y_r;
		}

		#if defined (CUOZBLAS) || defined (OZBLAS)
		print_info3 (&th, &ha);
		#else
		printf ("\n");
		#endif

		dim_dev_increment (&th);
		if (th.dim_m_const && th.dim_n_const) break;
	}
// --------------------------------------------
	delete[]hst_A;
	delete[]hst_X;
	delete[]hst_Y;
	delete[]hst_Y_t;

	#if defined (CUOZBLAS) || defined (OZBLAS)
	ozblasDestroy (&ha);
	#elif defined (CUBLAS)
	cublasDestroy (ha);
	#endif
// --------------------------------------------

	return 0;
}
