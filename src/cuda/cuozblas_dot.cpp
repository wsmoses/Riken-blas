#include "cuozblas_common.h"

template <typename TYPE1, typename TYPE2>
int32_t cuozblasRdot (
	cuozblasHandle_t *oh,
	const int32_t n,
	const TYPE1 *devA, const int32_t incx,
	const TYPE1 *devB, const int32_t incy,
	TYPE1 *ret
) {
	if (oh->reproModeFlag == 0 && oh->nSplitMax == 1) {
		blasRdot (oh->ch, n, devA, incx, devB, incy, ret);
		return 0;
	}
	if (incx != 1 || incy != 1 ) {
		fprintf (OUTPUT, "OzBLAS error: incx and incy are not supported.\n");
		exit (1);
	}

	TYPE1 fone = 1., fzero = 0.;

    cuozblasRgemm <TYPE1, TYPE2> (oh, 't', 'n', 1, 1, n, fone, devA, n, devB, n, fzero, ret, 1);

	//cuozblasRgemm <TYPE1, TYPE2> (oh, 't', 'n', 1, 1, n, fone, devA, n, devB, n, fzero, (TYPE1*)oh->devWorkCommon, 1);
	//cudaMemcpy (ret, oh->devWorkCommon, sizeof(TYPE1), cudaMemcpyDeviceToHost);
	return 0;
}
template int32_t cuozblasRdot <double, double> (cuozblasHandle_t *oh, const int32_t n, const double *devA, const int32_t incx, const double *devB, const int32_t incy, double *ret);
template int32_t cuozblasRdot <double, float> (cuozblasHandle_t *oh, const int32_t n, const double *devA, const int32_t incx, const double *devB, const int32_t incy, double *ret);
template int32_t cuozblasRdot <float, float> (cuozblasHandle_t *oh, const int32_t n, const float *devA, const int32_t incx, const float *devB, const int32_t incy, float *ret);
template int32_t cuozblasRdot <float, double> (cuozblasHandle_t *oh, const int32_t n, const float *devA, const int32_t incx, const float *devB, const int32_t incy, float *ret);

