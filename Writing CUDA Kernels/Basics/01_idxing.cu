#include <stdio.h>
#include <iostream>

__global__ void whoami(void)
{
    
    int block_id = blockIdx.x +
                   blockIdx.y * gridDim.x +
                   blockIdx.z * gridDim.x * gridDim.y;

    int block_offset = block_id *
                       blockDim.x * blockDim.y * blockDim.z;

    int thread_offset = threadIdx.x +
                        threadIdx.y * blockDim.x +
                        threadIdx.z * blockDim.x * blockDim.y;

    int id = block_offset + thread_offset;

    printf("%04d | Block (%d %d %d) = %3d | Thread (%d %d %d) = %3d\n",
           id,
           blockIdx.x, blockIdx.y, blockIdx.z, block_id,
           threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

int main(int argc, char **argv)
{
    const int b_x = 2, b_y = 3, b_z = 4;
    const int t_x = 4, t_y = 4, t_z = 4;

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;

    std::cout << blocks_per_grid << " blocks/grid" << std::endl;
    std::cout << threads_per_block << " threads/block" << std::endl;
    std::cout << threads_per_block * blocks_per_grid << " total threads" << std::endl;

    dim3 blocksPerGrid(b_x, b_y, b_z);
    dim3 threadsPerBlock(t_x, t_y, t_z);

    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}