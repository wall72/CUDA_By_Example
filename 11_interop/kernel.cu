
#include "common.h"
#include "device_launch_parameters.h"
#include "cpu_bitmap.h"
#include "cuda_gl_interop.h"

#define DIM 512

PFNGLBINDBUFFERARBPROC    glBindBuffer = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData = NULL;

GLuint bufferObj;
cudaGraphicsResource *resource;

__global__ void kernel(uchar4 *ptr) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float fx = x / (float)DIM - 0.5f;
	float fy = y / (float)DIM - 0.5f;
	unsigned char   green = 128 + 127 * sin(abs(fx * 100) - abs(fy * 100));

	ptr[offset].x = 0;
	ptr[offset].y = green;
	ptr[offset].z = 0;
	ptr[offset].w = 255;
}

static void key_func(unsigned char key, int x, int y) {
	switch (key) {
	case 27:
		HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &bufferObj);
		exit(0);
	}
}

static void draw_func(void) {
	glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}

int main(int argc, char **argv) {
	cudaDeviceProp prop;
	int dev;

	cudaMemset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 2;
	prop.minor = 0;
	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));

	printf("device id = %d\n", dev);

	HANDLE_ERROR(cudaSetDevice(dev));

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("bitmap");

	glBindBuffer = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
	glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
	glGenBuffers = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
	glBufferData = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);

	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone));

	uchar4 *devPtr;
	size_t size;
	HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));
	HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void **)&devPtr, &size, resource));

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel << <grids, threads >> >(devPtr);

	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));

	glutKeyboardFunc(key_func);
	glutDisplayFunc(draw_func);
	glutMainLoop();
}