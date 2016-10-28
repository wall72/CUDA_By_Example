
#include "common.h"
#include "device_launch_parameters.h"

#define N (33 * 1024)

__global__ void addKernel(int *a, int *b, int *c) {

	// ch04 block N thread 1
	// int tid = blockIdx.x;
	// ch05 block 1 thread N
	// int tid = threadIdx.x;
	// ch05 block (N+127)/128 thread 128
	// ch05 block 128 thread 128
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// if (tid < N) {
	//	   c[tid] = a[tid] + b[tid];
	// }

	// ch05 block 128 thread 128
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid = tid + blockDim.x * gridDim.x;
	}
}

int main(void) {

	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	HANDLE_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void **)&dev_c, N * sizeof(int)));

	for (int i = 0; i < N; i++) {
		a[i] = -1;
		b[i] = i * i;
	}

	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	// ch04 block N thread 1
	// addKernel<<<N, 1>>>(dev_a, dev_b, dev_c);
	// ch05 block 1 thread N
	// addKernel << <1, N>> >(dev_a, dev_b, dev_c);
	// ch05 block (N+127)/128 thread 128
	// addKernel << <(N + 127) / 128, 128>> >(dev_a, dev_b, dev_c);
	// ch05 block 128 thread 128
	addKernel << <128, 128 >> >(dev_a, dev_b, dev_c);

	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
