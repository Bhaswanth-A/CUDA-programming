/* 
Usage:
nvcc -arch=native -o julia 02_julia_set_viz.cu -lGL -lGLU -lglut
*/

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>
#include "../../common/cpu_bitmap.h"

#define DIM 1000

struct cuComplex
{
    float r;
    float i;

    __device__ cuComplex(float a, float b)
    {
        r = a;
        i = b;
    }

    __device__ float magnitude2(void)
    {
        return r * r - i * i;
    }

    __device__ cuComplex operator*(const cuComplex &a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    __device__ cuComplex operator+(const cuComplex &a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia(int x, int y)
{
    const float scale = 1.5; // for zoom
    // center complex plane around image plane by shifting by (DIM/2), and scale image coordinate by DIM/2
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    for (int i = 0; i < 200; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0; // If Z grows unbounded, the point does not belong to the set (black)
    }

    return 1; // If Z remains bounded, the point belongs to the set (red in this case)
}

__global__ void kernel(unsigned char *ptr)
{
    // map from threadIdx/blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;

    int offset = x + gridDim.x * y;

    // calculate value at pixel position
    int juliaValue = julia(x, y);
    // Since each pixel requires 4 bytes (for RGBA), the pixel at offset = N starts at index 4 * N in the ptr array
    ptr[offset * 4 + 0] = 255 * juliaValue; // Red channel (255 if in set, 0 if not)
    ptr[offset * 4 + 1] = 0;                // Green channel
    ptr[offset * 4 + 2] = 0;                // Blue channel
    ptr[offset * 4 + 3] = 255;              // Alpha
}

int main(void)
{
    CPUBitmap bitmap(DIM, DIM); // allocates memory for a DIM x DIM image
    size_t img_size = bitmap.image_size();

    unsigned char *dev_bitmap;

    cudaMalloc(&dev_bitmap, img_size);

    dim3 grid(DIM, DIM, 1);
    kernel<<<grid, 1>>>(dev_bitmap);

    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, img_size, cudaMemcpyDeviceToHost);

    bitmap.display_and_exit();

    cudaFree(dev_bitmap);
}
