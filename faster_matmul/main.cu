#include <bits/stdc++.h>
#include <cuda_runtime.h>

#define N 1024
#define TILE_SIZE 16

__global__ void TiledMatmul(float *A, float *B, float *C)
{
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float sum = 0;

    for (int phase = 0; phase < N / TILE_SIZE; phase++)
    {
        // Load tile of A and B
        s_A[ty][tx] = A[row * N + phase * TILE_SIZE + tx];
        s_B[ty][tx] = B[(phase * TILE_SIZE + ty) * N + col];
        __syncthreads();
        // Create partial sum
        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}

int main()
{
    float *A;
    float *B;
    float *C;
    cudaMallocManaged(&A, N * N * sizeof(float));
    cudaMallocManaged(&B, N * N * sizeof(float));
    cudaMallocManaged(&C, N * N * sizeof(float));

    // Initialize A and B
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = i;
            B[i * N + j] = j;
        }
    }

    dim3 block(16, 16);
    dim3 grid(ceil(N / block.x), ceil(N / block.y));
    TiledMatmul<<<grid, block>>>(A, B, C);

    cudaDeviceSynchronize();

    printf("Output:\n");
    printf("1, 1 = %f\n", C[0]);
    printf("1, 2 = %f\n", C[1]);
    printf("2, 3 = %f\n", C[N + 2]);
}