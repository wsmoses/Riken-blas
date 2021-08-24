#define ROUTINE_FLOPS(m,n,k)	((double)2.*n)
#define ROUTINE_BYTES(m,n,k,s)	((double)2.*n*s)
#include "testing_common.h"
#include "testing_common.cpp"

int32_t
main (int32_t argc, char **argv)
{
	int32_t i;
	double t0, t1, sec;
	mpreal hst_result_r;
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
	FP_TYPE hst_result_t = 0.;
	#if defined (CUDA)
	FP_TYPE *dev_X, *dev_Y;
	#endif

	// dim setup
	dim_hst_setup (&th);
	// malloc host memory
	FP_TYPE *hst_X = new FP_TYPE[th.dim_n_hst * th.incx];
	FP_TYPE *hst_Y = new FP_TYPE[th.dim_n_hst * th.incy];
	// initialize (1:val, 2:phi, 3:erange)
	if (th.trunc != 0)  // for reduced-precision performance evaluation
		printf ("### !!! Truncated inputs !!!\n");
	mublasInitMat (&th, th.dim_n_hst*th.incx, 1, 0, hst_X, th.phi, 3, th.trunc);
	mublasInitMat (&th, th.dim_n_hst*th.incy, 1, 0, hst_Y, th.phi, 3, th.trunc); 
// --------------------------------------------

	print_info2 (&th);

// evaluation ---------------------------------
	dim_dev_setup (&th);
	while (1) {
		if (th.dim_n_const == 0 && th.dim_n_dev > th.dim_stop) break;

		#if defined (CUDA)
		int32_t sizeType = sizeof (FP_TYPE);
		cudaMalloc ((void **) &dev_X, sizeType * th.dim_n_dev * th.incx);
		cudaMalloc ((void **) &dev_Y, sizeType * th.dim_n_dev * th.incy);
		//cublasSetVector(th.dim_n_dev * th.incx, sizeType, hst_X, 1, dev_X, 1);
		//cublasSetVector(th.dim_n_dev * th.incy, sizeType, hst_Y, 1, dev_Y, 1);
		#endif

		printf ("--\t%d\t--", th.dim_n_dev);
		get_routine_theoretial_performance (&th);
		// execution ---------------------------------
		if (th.mode == 'p') {
			for (i = 0; i < WLOOP; i++) { // warm up
				#if defined (CUBLAS) || defined (CUOZBLAS) 
				trgRdot (ha, th.dim_n_dev, dev_X, 1, dev_Y, 1, &hst_result_t);
				#else
				hst_result_t = trgRdot (ha, th.dim_n_dev, hst_X, 1, hst_Y, 1);
				#endif
			}
			t0 = gettime ();
			for (i = 0; i < NLOOP; i++) {
				#if defined (CUBLAS) || defined (CUOZBLAS) 
				trgRdot (ha, th.dim_n_dev, dev_X, 1, dev_Y, 1, &hst_result_t); 
				#else
				hst_result_t = trgRdot (ha, th.dim_n_dev, hst_X, 1, hst_Y, 1);
				#endif
			}
			t1 = gettime ();
			sec = (t1 - t0) / NLOOP;
			printf ("\t%1.3e\t%1.3e\t%1.3e\t%d", sec, (th.routine_flops / sec) * 1.0e-9, (th.routine_bytes / sec) * 1.0e-9, NLOOP);
		}
		if (th.mode == 'c') {
			#if defined (CUBLAS) || defined (CUOZBLAS) 
			trgRdot (ha, th.dim_n_dev, dev_X, 1, dev_Y, 1, &hst_result_t); 
			#else
			hst_result_t = trgRdot (ha, th.dim_n_dev, hst_X, 1, hst_Y, 1);
			#endif
		}
		// -------------------------------------------

		#if defined (CUDA)
		cudaFree (dev_X);
		cudaFree (dev_Y);
		#endif

		if (th.mode == 'c') {
			mpreal *hst_X_r = new mpreal[th.dim_n_hst];
			mpreal *hst_Y_r = new mpreal[th.dim_n_hst];
			mublasConvMat (th.dim_n_hst*th.incx, 1, hst_X, 0, hst_X_r, 0);
			mublasConvMat (th.dim_n_hst*th.incy, 1, hst_Y, 0, hst_Y_r, 0);
			hst_result_r = refRdot (th.dim_n_dev,hst_X_r,1,hst_Y_r,1);
			mublasCheckMatrix (1, 1, &hst_result_t, 1, &hst_result_r, 1);
			delete[]hst_X_r;
			delete[]hst_Y_r;
		}

		#if defined (CUOZBLAS) || defined (OZBLAS)
		print_info3 (&th, &ha);
		#else
		printf ("\n");
		#endif

		dim_dev_increment (&th);
		if (th.dim_m_const && th.dim_n_const && th.dim_k_const) break;
	}
// --------------------------------------------

// shutdown -----------------------------------
	delete[]hst_X;
	delete[]hst_Y;

	#if defined (CUOZBLAS) || defined (OZBLAS)
	ozblasDestroy (&ha);
	#elif defined (CUBLAS)
	cublasDestroy (ha);
	#endif
// --------------------------------------------

	return 0;
}
