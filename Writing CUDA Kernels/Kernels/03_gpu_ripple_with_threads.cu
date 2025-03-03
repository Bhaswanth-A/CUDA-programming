#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>
#include "../../common/cpu_bitmap.h"
#include "../../common/cpu_anim.h"

#define DIM 1000

struct  DataBlock
{
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;

};

// clean up memory allocated on GPU
void cleanup(DataBlock *d){
    cudaFree(d->dev_bitmap);
}

__global__ void kernel(unsigned char *ptr, int ticks){
    // map from threadIdx/blockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float fx = DIM/2 - x;
    float fy = DIM/2 - y;
    float d = sqrt(fx*fx + fy*fy);

    unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d/10.0f - ticks/7.0f) / (d/10.0f + 1.0f));

    ptr[offset*4 + 0] = grey;
    ptr[offset*4 + 1] = grey;
    ptr[offset*4 + 2] = 127;
    ptr[offset*4 + 3] = 255;
}

void generate_frame(DataBlock *d, int ticks){
    dim3 blocksPerGrid(DIM/16, DIM/16, 1);
    dim3 threadsPerBlock(16,16,1);

    kernel<<<blocksPerGrid, threadsPerBlock>>>(d->dev_bitmap, ticks);

    cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost);
}

int main(void){
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;
    size_t img_size = bitmap.image_size();

    cudaMalloc(&data.dev_bitmap, img_size);

    bitmap.anim_and_exit((void(*)(void*,int))generate_frame, (void(*)(void*))cleanup);
}
