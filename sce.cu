#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <cfloat>
#include <curand_kernel.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>


static void
CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString (err) << "(" << err << ") at " << file << ":" << line
			<< std::endl;
	exit (EXIT_FAILURE);
}
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


void fread_with_check(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
	size_t nmemb_read = fread(ptr, size, nmemb, stream);
	if (nmemb_read == nmemb)
		return;
	printf("fread error");
	exit (EXIT_FAILURE);
}


#define MAX(a,b) ( (a) > (b) ? (a) : (b) )
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )

#ifndef DIM
#define DIM 2
#endif

#define alpha0 0.5

typedef float real;
typedef long long longint;

int blockSize, blockCount, nWorker;

longint *I, *J;
double *P, *weights;
real *Y;
longint nn, ne, maxIter, nRepuSamp;
real eta0;
gsl_ran_discrete_t *gsl_de, *gsl_dn;
int bConstantEta;

longint *d_I, *d_J;
real *d_Y;
real *d_qsum;
real *d_qcount;
curandState *d_nnStates1, *d_nnStates2;
curandState *d_neStates;
gsl_ran_discrete_t *d_gsl_de, *d_gsl_dn;
double *d_gsl_de_F, *d_gsl_dn_F;
size_t *d_gsl_de_A, *d_gsl_dn_A;

real *d_Eq;
real *d_qsum_total;
real *d_qcount_total;

real alpha;

void
loadP (const char *fnameP, int bBinaryInput)
{
	FILE *fpP = fopen (fnameP, "r");
	if (bBinaryInput)
	{
		fread_with_check (&nn, sizeof(longint), 1, fpP);
		fread_with_check (&ne, sizeof(longint), 1, fpP);
		I = new longint[ne];
		J = new longint[ne];
		P = new double[ne];
		fread_with_check (I, sizeof(longint), ne, fpP);
		fread_with_check (J, sizeof(longint), ne, fpP);
		fread_with_check (P, sizeof(double), ne, fpP);
	}
	else
	{
		if (fscanf (fpP, "%lld %lld", &nn, &ne) != 2)
		{
			printf("Error in reading nn or ne!\n");
			exit(EXIT_FAILURE);
		}
		I = new longint[ne];
		J = new longint[ne];
		P = new double[ne];
		for (longint e = 0; e < ne; e++)
			if (fscanf (fpP, "%lld %lld %lg", I + e, J + e, P + e) != 3)
			{
				printf("Error in reading I, J, or P!\n");
				exit(EXIT_FAILURE);
			}
	}
	fclose (fpP);
}

void
loadWeights (const char *fnameWeights, int bBinaryInput)
{
	weights = new double[nn];
	if (strcmp (fnameWeights, "none") == 0)
	{
		for (longint i = 0; i < nn; i++)
			weights[i] = 1.0;
	}
	else
	{
		FILE *fpWeights = fopen (fnameWeights, "r");
		if (bBinaryInput)
			fread_with_check (weights, sizeof(double), nn, fpWeights);
		else
			for (longint i = 0; i < nn; i++)
				if (fscanf (fpWeights, "%lg", weights + i) != 1)
				{
					printf("Error in reading weights!\n");
					exit(EXIT_FAILURE);
				}
		fclose (fpWeights);
	}
}

void
loadY0 (const char *fnameY0, int bBinaryInput)
{
	Y = new real[nn * DIM];
	if (strcmp (fnameY0, "none") == 0)
	{
		srand (0);
		for (longint i = 0; i < nn; i++)
			for (longint d = 0; d < DIM; d++)
				Y[d + i * DIM] = rand () * 1e-4 / RAND_MAX;
	}
	else
	{
		FILE *fpY0 = fopen (fnameY0, "r");
		if (bBinaryInput)
			fread_with_check (Y, sizeof(real), nn * DIM, fpY0);
		else
			for (longint i = 0; i < nn; i++)
				if (fscanf (fpY0, "%f %f", Y + i * DIM, Y + i * DIM + 1) != 2)
				{
					printf("Error in reading Y0!\n");
					exit(EXIT_FAILURE);
				}
		fclose (fpY0);
	}
}

void
saveY (const char* fnameY)
{
	FILE *fpY = fopen (fnameY, "w+");
	for (longint i = 0; i < nn; i++)
	{
		for (longint d = 0; d < DIM; d++)
		{
			fprintf (fpY, "%.6f", Y[d + i * DIM]);
			if (d < DIM - 1)
				fprintf (fpY, " ");
		}
		fprintf (fpY, "\n");
	}

	fclose (fpY);
}

void
freeMemory ()
{
	delete[] I;
	delete[] J;
	delete[] P;
	delete[] Y;
	delete[] weights;
}

void
allocateDataAndCopy2Device ()
{
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_Y, sizeof(real)*nn*DIM));
	CUDA_CHECK_RETURN(cudaMalloc ((void** )&d_I, sizeof(longint) * ne));
	CUDA_CHECK_RETURN(cudaMalloc ((void** )&d_J, sizeof(longint) * ne));

	CUDA_CHECK_RETURN(cudaMemcpy(d_Y, Y, sizeof(real)*nn*DIM, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy (d_I, I, sizeof(longint) * ne, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy (d_J, J, sizeof(longint) * ne, cudaMemcpyHostToDevice));

	real Eq = 1;
	CUDA_CHECK_RETURN(cudaMalloc ((void** )&d_Eq, sizeof(real)));
	CUDA_CHECK_RETURN(cudaMemcpy (d_Eq, &Eq, sizeof(real), cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc ((void ** )&d_qsum, nWorker * sizeof(real)));
	CUDA_CHECK_RETURN(cudaMalloc ((void ** )&d_qcount, nWorker * sizeof(real)));
	CUDA_CHECK_RETURN(cudaMalloc ((void ** )&d_qsum_total, sizeof(real)));
	CUDA_CHECK_RETURN(cudaMalloc ((void ** )&d_qcount_total, sizeof(real)));

}

void
freeDataInDevice ()
{
	CUDA_CHECK_RETURN(cudaFree (d_Y));
	CUDA_CHECK_RETURN(cudaFree (d_I));
	CUDA_CHECK_RETURN(cudaFree (d_J));
	CUDA_CHECK_RETURN(cudaFree (d_qsum));
	CUDA_CHECK_RETURN(cudaFree (d_qcount));

	CUDA_CHECK_RETURN(cudaFree (d_nnStates1));
	CUDA_CHECK_RETURN(cudaFree (d_nnStates2));
	CUDA_CHECK_RETURN(cudaFree (d_neStates));
	CUDA_CHECK_RETURN(cudaFree (d_gsl_de_A));
	CUDA_CHECK_RETURN(cudaFree (d_gsl_de_F));
	CUDA_CHECK_RETURN(cudaFree (d_gsl_de));
	CUDA_CHECK_RETURN(cudaFree (d_gsl_dn_A));
	CUDA_CHECK_RETURN(cudaFree (d_gsl_dn_F));
	CUDA_CHECK_RETURN(cudaFree (d_gsl_dn));

	CUDA_CHECK_RETURN(cudaFree (d_qsum_total));
	CUDA_CHECK_RETURN(cudaFree (d_qcount_total));
}

__device__ size_t
my_curand_discrete (curandState *state, const gsl_ran_discrete_t *g)
{
	size_t c = 0;
	double u, f;
	u = curand_uniform (state);
	c = (u * (g->K));
	f = (g->F)[c];
	if (f == 1.0)
		return c;

	if (u < f)
	{
		return c;
	}
	else
	{
		return (g->A)[c];
	}
}

__global__ void
setupCURANDKernel (curandState *nnStates1, curandState *nnStates2, curandState *neStates)
{
	longint workerIdx = (longint) (blockIdx.x * blockDim.x + threadIdx.x);
	curand_init (314159, /* the seed */
				 workerIdx, /* the sequence number */
				 0, /* not use the offset */
				 &nnStates1[workerIdx]);
	curand_init (314159 + 1, /* the seed */
				 workerIdx, /* the sequence number */
				 0, /* not use the offset */
				 &nnStates2[workerIdx]);
	curand_init (271828, /* the seed */
				 workerIdx, /* the sequence number */
				 0, /* not use the offset */
				 &neStates[workerIdx]);
}

__global__ void
assembleGSLKernel (gsl_ran_discrete_t *d_gsl_de, size_t *d_gsl_de_A, double *d_gsl_de_F, gsl_ran_discrete_t *d_gsl_dn,
				   size_t *d_gsl_dn_A, double *d_gsl_dn_F)
{
	d_gsl_de->A = d_gsl_de_A;
	d_gsl_de->F = d_gsl_de_F;
	d_gsl_dn->A = d_gsl_dn_A;
	d_gsl_dn->F = d_gsl_dn_F;
}

void
setupDiscreteDistribution ()
{
	CUDA_CHECK_RETURN(cudaMalloc ((void ** )&d_nnStates1, blockCount * blockSize * sizeof(curandState)));
	CUDA_CHECK_RETURN(cudaMalloc ((void ** )&d_nnStates2, blockCount * blockSize * sizeof(curandState)));
	CUDA_CHECK_RETURN(cudaMalloc ((void ** )&d_neStates, blockCount * blockSize * sizeof(curandState)));
	setupCURANDKernel <<<blockCount, blockSize>>> (d_nnStates1, d_nnStates2, d_neStates);

	gsl_rng_env_setup ();
	gsl_de = gsl_ran_discrete_preproc (ne, P);
	gsl_dn = gsl_ran_discrete_preproc (nn, weights);
	CUDA_CHECK_RETURN(cudaMalloc ((void ** )&d_gsl_de, sizeof(gsl_ran_discrete_t)));
	CUDA_CHECK_RETURN(cudaMalloc ((void ** )&d_gsl_dn, sizeof(gsl_ran_discrete_t)));
	CUDA_CHECK_RETURN(cudaMalloc ((void ** )&d_gsl_de_A, sizeof(size_t) * ne));
	CUDA_CHECK_RETURN(cudaMalloc ((void ** )&d_gsl_de_F, sizeof(double) * ne));
	CUDA_CHECK_RETURN(cudaMalloc ((void ** )&d_gsl_dn_A, sizeof(size_t) * nn));
	CUDA_CHECK_RETURN(cudaMalloc ((void ** )&d_gsl_dn_F, sizeof(double) * nn));
	CUDA_CHECK_RETURN(cudaMemcpy (d_gsl_de, gsl_de, sizeof(gsl_ran_discrete_t), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy (d_gsl_de_A, gsl_de->A, sizeof(size_t) * ne, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy (d_gsl_de_F, gsl_de->F, sizeof(double) * ne, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy (d_gsl_dn, gsl_dn, sizeof(gsl_ran_discrete_t), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy (d_gsl_dn_A, gsl_dn->A, sizeof(size_t) * nn, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy (d_gsl_dn_F, gsl_dn->F, sizeof(double) * nn, cudaMemcpyHostToDevice));
	assembleGSLKernel <<<1, 1>>> (d_gsl_de, d_gsl_de_A, d_gsl_de_F, d_gsl_dn, d_gsl_dn_A, d_gsl_dn_F);
	gsl_ran_discrete_free (gsl_de);
	gsl_ran_discrete_free (gsl_dn);
}

__global__ void
sceUpdateYKernel (curandState *nnStates1, curandState *nnStates2, curandState *neStates, gsl_ran_discrete_t* d_gsl_dn,
					gsl_ran_discrete_t* d_gsl_de, real *Y, longint *I, longint *J, real *d_Eq, real *qsum, real *qcount,
					longint nn, longint ne, real eta, longint nRepuSamp, real nsq, real attrCoef, real alpha)
{
	int workerIdx = blockIdx.x * blockDim.x + threadIdx.x;
	real dY[DIM];
	real c = 1.0 / ((*d_Eq) * nsq);
	qsum[workerIdx] = 0.0;
	qcount[workerIdx] = 0.0;

	real repuCoef = 2 * c / nRepuSamp * nsq;
	for (longint r = 0; r < nRepuSamp + 1; r++)
	{
		longint k, l;
		if (r == 0)
		{
			longint e = (longint) (my_curand_discrete (neStates + workerIdx, d_gsl_de) % ne);
			k = I[e];
			l = J[e];
		}
		else
		{
			k = (longint) (my_curand_discrete (nnStates1 + workerIdx, d_gsl_dn) % nn);
			l = (longint) (my_curand_discrete (nnStates2 + workerIdx, d_gsl_dn) % nn);
		}

		if (k == l)
			continue;

		longint lk = k * DIM;
		longint ll = l * DIM;
		real dist2 = 0.0;
		for (longint d = 0; d < DIM; d++)
		{
			dY[d] = Y[d + lk] - Y[d + ll];
			dist2 += dY[d] * dY[d];
		}
		real q = 1.0 / (1 + dist2);

		real g;
		if (r == 0)
			g = -attrCoef * q;
		else
			g = repuCoef * q * q;

		for (longint d = 0; d < DIM; d++)
		{
			real gain = eta * g * dY[d];
			Y[d + lk] += gain;
			Y[d + ll] -= gain;

		}
		qsum[workerIdx] += r==0 ? alpha * q : (1-alpha) * q;
		qcount[workerIdx] += r==0 ? alpha : (1-alpha);
	}
}

__global__ void
resetQsumQCountTotalKernel (real *d_qsum_total, real *d_qcount_total)
{
	(*d_qsum_total) = 0.0;
	(*d_qcount_total) = 0;
}

template<typename T>
	__global__ void
	reduceSumArrayKernel (T *array, int n, T* arraySum)
	{
		T sum = 0;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		{
			sum += array[i];
		}
		atomicAdd (arraySum, sum);
	}

__global__ void
updateEqKernel (real *d_Eq, real *d_qsum_total, real* d_qcount_total, real nsq)
{
	(*d_Eq) = ((*d_Eq) * nsq + (*d_qsum_total)) / (nsq + (*d_qcount_total));
}

void
sce ()
{
	//cudaSetDevice (0);
	cudaDeviceReset ();

	nWorker = blockSize * blockCount;
	real nsq = (real) nn * (nn - 1);

	real Psum = 0.0;
	for (longint e = 0; e < ne; e++)
		Psum += P[e];
	for (longint e = 0; e < ne; e++)
		P[e] /= Psum;

	real wsum = 0.0;
	for (longint i=0; i<nn; i++)
		wsum += weights[i];
	for (longint i=0; i<nn; i++)
		weights[i] /= wsum;

	allocateDataAndCopy2Device ();

	setupDiscreteDistribution ();


	for (longint iter = 0; iter < maxIter; iter++)
	{
		real eta;
		if (bConstantEta)
			eta = eta0;
		else {
			eta = eta0 * (1 - (real) iter / (maxIter - 1));
			eta = MAX(eta, eta0 * 1e-4);
		}

		real alpha_effective = alpha;

		real attrCoef = 2;
		sceUpdateYKernel <<<blockCount, blockSize>>> (d_nnStates1, d_nnStates2, d_neStates, d_gsl_dn, d_gsl_de, d_Y,
														d_I, d_J, d_Eq, d_qsum, d_qcount, nn, ne, eta, nRepuSamp, nsq,
														attrCoef, alpha_effective);

		resetQsumQCountTotalKernel <<<1, 1>>> (d_qsum_total, d_qcount_total);
		reduceSumArrayKernel <<<16, 128>>> (d_qsum, nWorker, d_qsum_total);
		reduceSumArrayKernel <<<16, 128>>> (d_qcount, nWorker, d_qcount_total);
		updateEqKernel <<<1, 1>>> (d_Eq, d_qsum_total, d_qcount_total, nsq);

		if (iter % MAX(1, maxIter / 1000) == 0)
		{
			printf ("%cOptimizing progress: %.3lf%%", 13, (real) iter / (real) maxIter * 100);
			fflush (stdout);
		}
	}

	CUDA_CHECK_RETURN(cudaMemcpy(Y, d_Y, sizeof(real)*nn*DIM, cudaMemcpyDeviceToHost));

	freeDataInDevice ();
}


int
main (int argc, char **argv)
{
	printf ("Usage: sce bBinaryInput P_file Y_file weights_file Y0_file maxIter eta0 nRepuSamp blockSize blockCount alpha bConstantEta\n");
	int bBinaryInput = atoi(argv[1]);
	const char *fnameP = argv[2];
	const char *fnameY = argv[3];
	const char *fnameWeights = argv[4];
	const char *fnameY0 = argv[5];
	maxIter = atoi (argv[6]);
	eta0 = atof (argv[7]);
	nRepuSamp = atoi (argv[8]);
	blockSize = atoi (argv[9]);
	blockCount = atoi (argv[10]);
	alpha = atof(argv[11]);
	bConstantEta = atoi(argv[12]);

	printf ("maxIter=%lld, eta0=%f, nRepuSamp=%lld, blockSize=%d, blockCount=%d, bConstantEta=%d\n", maxIter, eta0, nRepuSamp,
			blockSize, blockCount, bConstantEta);

	loadP (fnameP, bBinaryInput);
	loadWeights (fnameWeights, bBinaryInput);
	loadY0 (fnameY0, bBinaryInput);

	clock_t start = clock ();
	sce ();
	clock_t end = clock ();
	float seconds = (float) (end - start) / CLOCKS_PER_SEC;
	printf ("\nSCE used %.2f seconds\n", seconds);

	saveY (fnameY);

	freeMemory ();

	printf ("Done.\n");

	return 0;
}

