
#include "common.h"
#include "device_launch_parameters.h"

#define imin(a, b) (a < b ? a : b)
#define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

struct Lock {
	int *mutex;

	Lock(void) {
		HANDLE_ERROR(cudaMalloc((void **)&mutex, sizeof(int)));
		HANDLE_ERROR(cudaMemset(mutex, 0, sizeof(int)));
	}

	//~Lock(void) {
	//	HANDLE_ERROR(cudaFree(mutex));
	//}

	__device__ void lock(void) {
		while (atomicCAS(mutex, 0, 1) != 0);
		//__threadfence();
	}

	__device__ void unlock(void) {
		//__threadfence();
		atomicExch(mutex, 0);
	}
};

__global__ void dot(Lock lock, float *a, float *b, float *c) {
	__shared__ float cache[threadsPerBlock];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float temp = 0;
	while (tid < N) {
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	cache[cacheIndex] = temp;

	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0) {
		lock.lock();
		*c += cache[0];
		lock.unlock();
	}
}

int main(void) {
	float *a, *b, c = 0;
	float *dev_a, *dev_b, *dev_c;

	a = (float *)malloc(N * sizeof(float));
	b = (float *)malloc(N * sizeof(float));

	HANDLE_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void **)&dev_c, sizeof(float)));

	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i * 2;
	}

	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_c, &c, sizeof(float), cudaMemcpyHostToDevice));

	Lock lock;
	dot << <blocksPerGrid, threadsPerBlock >> >(lock, dev_a, dev_b, dev_c);

	HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(float), cudaMemcpyDeviceToHost));

	printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float)(N - 1)));

	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));

	free(a);
	free(b);
}