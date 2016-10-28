
#include "common.h"
#include "device_launch_parameters.h"

#define imin(a, b) (a < b ? a : b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(int size, float *a, float *b, float *c) {
	__shared__ float cache[threadsPerBlock];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float temp = 0;
	while (tid < size) {
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

	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

float malloc_test(int size) {
	cudaEvent_t start, stop;
	float elapsedTime;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;

	a = (float *)malloc(N * sizeof(float));
	b = (float *)malloc(N * sizeof(float));
	partial_c = (float *)malloc(blocksPerGrid * sizeof(float));
	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i * 2;
	}

	HANDLE_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void **)&dev_partial_c, blocksPerGrid * sizeof(float)));

	HANDLE_ERROR(cudaEventRecord(start, 0));

	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

	dot << <blocksPerGrid, threadsPerBlock >> >(size, dev_a, dev_b, dev_partial_c);

	HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	c = 0;
	for (int i = 0; i < blocksPerGrid; i++) {
		c += partial_c[i];
	}
	printf("Value calculated:  %f\n", c);

	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_partial_c));

	free(a);
	free(b);
	free(partial_c);

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	return elapsedTime;
}

float host_alloc_test(int size) {
	cudaEvent_t start, stop;
	float elapsedTime;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;

	HANDLE_ERROR(cudaHostAlloc((void **)&a, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void **)&b, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void **)&partial_c, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped));
	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i * 2;
	}

	HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a, a, 0));
	HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b, b, 0));
	HANDLE_ERROR(cudaHostGetDevicePointer(&dev_partial_c, partial_c, 0));

	HANDLE_ERROR(cudaEventRecord(start, 0));

	dot << <blocksPerGrid, threadsPerBlock >> >(size, dev_a, dev_b, dev_partial_c);

	HANDLE_ERROR(cudaThreadSynchronize());

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	c = 0;
	for (int i = 0; i < blocksPerGrid; i++) {
		c += partial_c[i];
	}
	printf("Value calculated:  %f\n", c);

	HANDLE_ERROR(cudaFreeHost(a));
	HANDLE_ERROR(cudaFreeHost(b));
	HANDLE_ERROR(cudaFreeHost(partial_c));

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	return elapsedTime;
}

int main(void) {
	cudaDeviceProp prop;
	int whichDevice;
	HANDLE_ERROR(cudaGetDevice(&whichDevice));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));

	if (prop.integrated == 1)
		printf("Integrated device\n");
	else
		printf("Discrete device\n");

	if (prop.canMapHostMemory != 1) {
		printf("Device can not map memory.\n");
		return 0;
	}

	HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

	float elapsedTime;

	elapsedTime = malloc_test(N);
	printf("Time using cudaMalloc: %3.1f ms\n", elapsedTime);

	elapsedTime = host_alloc_test(N);
	printf("Time using cudaHostAlloc: %3.1f ms\n", elapsedTime);

	return 0;
}