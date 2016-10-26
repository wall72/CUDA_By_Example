
#include "common.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

int main()
{
	cudaDeviceProp prop;

	int count = 0;

	HANDLE_ERROR(cudaGetDeviceCount(&count));

	for (int i = 0; i < count; i++) {
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));

		printf(" --- General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);

		printf("Device copy overlap: ");
		if (prop.deviceOverlap)
			printf("Enabled\n");
		else
			printf("Disabled\n");

		printf("Kernel execution timeout: ");
		if (prop.kernelExecTimeoutEnabled)
			printf("Enabled\n");
		else
			printf("Disabled\n");

		printf(" --- Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		printf("Total constant mem: %ld\n", prop.totalConstMem);
		printf("Max mem pitch: %ld\n", prop.memPitch);
		printf("Texture alignment: %ld\n", prop.textureAlignment);

		printf(" --- MP Information for device %d ---\n", i);
		printf("Multiprocessor count: %d\n", prop.multiProcessorCount);

		printf("CUDA Cores/MP count: %d\n", _ConvertSMVer2Cores(prop.major, prop.minor) * prop.multiProcessorCount);


		printf("Thread in warp: %d\n", prop.warpSize);

		printf("Shared mem per block: %d\n", prop.sharedMemPerBlock);
		printf("Registers per block: %d\n", prop.regsPerBlock);

		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max threads dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	}

	return 0;
}
