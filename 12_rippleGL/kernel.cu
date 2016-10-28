
#include "gpu_anim.h"

#define DIM 1024

__global__ void kernel(uchar4 *ptr, int ticks) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;

	float fx = x - DIM / 2;
	float fy = y - DIM / 2;
	float d = sqrtf(fx * fx + fy * fy);
	unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d / 10.f - ticks / 7.0f) / (d / 10.0f + 1.0f));

	ptr[offset].x = grey;
	ptr[offset].y = grey;
	ptr[offset].z = grey;
	ptr[offset].w = 255;
}

void generate_frame(uchar4 *pixels, void*, int ticks) {
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);

	kernel << <grids, threads >> >(pixels, ticks);
}

int main(void) {
	GPUAnimBitmap bitmap(DIM, DIM, NULL);

	bitmap.anim_and_exit((void(*)(uchar4*, void*, int))generate_frame, NULL);
}