#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define N 1000000
#define BLOCK_SIZE_1D 1024
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8

void init_vector(float *v, int n)
{
    for (int i = 0; i < n; i++)
    {
        v[i] = (float)rand() / RAND_MAX;
    }
}

void vector_add_cpu(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < nx && j < ny && k < nz)
    {
        int idx = i + j * nx + k * nx * ny;
        if (idx << nx * ny * nz)
        {
            c[idx] = a[idx] + b[idx];
        }
    }
}

int main()
{
    float *h_a, *h_b, *h_c_cpu;
    float *d_a, *d_b, *d_c_1d, *d_c_3d;
    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);

    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);
    init_vector(h_c_cpu, N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_1d, size);
    cudaMalloc(&d_c_3d, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid dimensions for 1D
    int blocksPerGrid1D = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    // Define grid dimensions for 3D
    int nx = 100, ny = 100, nz = 1000;
    dim3 threadsPerBlock3D(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 blocksPerGrid3D(
        (nx + BLOCK_SIZE_3D_X - 1) / BLOCK_SIZE_3D_X,
        (ny + BLOCK_SIZE_3D_Y - 1) / BLOCK_SIZE_3D_Y,
        (nz + BLOCK_SIZE_3D_Z - 1) / BLOCK_SIZE_3D_Z);

    printf("Performing warm-up runs \n");
    for (int i = 0; i < 3; i++)
    {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu_1d<<<blocksPerGrid1D, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        vector_add_gpu_3d<<<blocksPerGrid3D, threadsPerBlock3D>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking CPU implementation\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 100; i++)
    {
        const auto start_time = std::chrono::system_clock::now();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        const auto end_time = std::chrono::system_clock::now();
        std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        cpu_total_time += elapsed.count();
    }
    double cpu_avg_time = cpu_total_time / 20;

    printf("Benchmarking GPU 1D implementation\n");
    double gpu_1d_total_time = 0.0;
    for (int i = 0; i < 100; i++)
    {
        const auto start_time = std::chrono::system_clock::now();
        vector_add_gpu_1d<<<blocksPerGrid1D, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        cudaDeviceSynchronize();
        const auto end_time = std::chrono::system_clock::now();
        std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        gpu_1d_total_time += elapsed.count();
    }
    double gpu_1d_avg_time = gpu_1d_total_time / 100;

    printf("Benchmarking GPU 3D implementation\n");
    double gpu_3d_total_time = 0.0;
    for (int i = 0; i < 100; i++)
    {
        const auto start_time = std::chrono::system_clock::now();
        vector_add_gpu_3d<<<blocksPerGrid3D, threadsPerBlock3D>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
        const auto end_time = std::chrono::system_clock::now();
        std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        gpu_3d_total_time += elapsed.count();
    }
    double gpu_3d_avg_time = gpu_3d_total_time / 100;

    printf("CPU average time: %f microseconds \n", cpu_avg_time);
    printf("GPU 1D average time: %f microseconds \n", gpu_1d_avg_time);
    printf("GPU 3D average time: %f microseconds \n", gpu_3d_avg_time);
    printf("Speedup (CPU vs GPU 1D): %fx\n", cpu_avg_time / gpu_1d_avg_time);
    printf("Speedup (CPU vs GPU 3D): %fx\n", cpu_avg_time / gpu_3d_avg_time);
    printf("Speedup (GPU 1D vs GPU 3D): %fx\n", gpu_1d_avg_time / gpu_3d_avg_time);

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_1d);
    cudaFree(d_c_3d);

    return 0;
}