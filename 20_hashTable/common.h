#ifndef __COMMON_H__
#define __COMMON_H__

#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define HANDLE_NULL(a) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

void* big_random_block(int size) {
	unsigned char *data = (unsigned char*)malloc(size);
	HANDLE_NULL(data);
	for (int i = 0; i<size; i++)
		data[i] = rand();

	return data;
}

int* big_random_block_int(int size) {
	int *data = (int*)malloc(size * sizeof(int));
	HANDLE_NULL(data);
	for (int i = 0; i<size; i++)
		data[i] = rand();

	return data;
}

#endif