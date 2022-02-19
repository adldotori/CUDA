#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;

#define N_FEATURE 1024
#define N_FIELD 16
#define DIM_FEAT 4
#define INPUT_CNT 1024

__global__ void FwFM(float *result, float *feature, int *input_data, float *interaction_weight)
{
    int data_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // only consider second order interaction.
    float res = 0;
    for (int i = 0; i < N_FIELD; i++)
    {
        for (int j = i + 1; j < N_FIELD; j++)
        {
            int didx = input_data[data_idx * N_FIELD + i];
            int djdx = input_data[data_idx * N_FIELD + j];

            float sum = 0;
            for (int k = 0; k < DIM_FEAT; k++)
            {
                sum += feature[didx * DIM_FEAT + k] * feature[djdx * DIM_FEAT + k];
            }
            res += interaction_weight[i * N_FIELD + j] * sum;
        }
    }
    result[data_idx] = res;
    // int local_input_data[N_FIELD];
    // float local_feature[N_FIELD][DIM_FEAT];

    // int bx = blockIdx.x;
    // int by = blockIdx.y;
    // int tx = threadIdx.x;
    // int ty = threadIdx.y;
    // int col = bx * blockDim.x + tx;
    // int row = by * blockDim.y + ty;

    // for (int i = 0; i < N_FIELD; i++)
    // {
    //     local_input_data[i] = input_data[col * N_FIELD + i];
    //     for (int j = 0; j < DIM_FEAT; j++)
    //         local_feature[i][j] = feature[local_input_data[i] * DIM_FEAT + j];
    // }

    // // only consider second order interaction.
    // float res = 0;
    // for (int i = 0; i < N_FIELD; i++)
    // {
    //     for (int j = i + 1; j < N_FIELD; j++)
    //     {
    //         float sum = 0;
    //         for (int k = 0; k < DIM_FEAT; k++)
    //         {
    //             sum += local_feature[i][k] * local_feature[j][k];
    //         }
    //         res += interaction_weight[i * N_FIELD + j] * sum;
    //     }
    // }
    // result[col] = res;
}

float rand_float()
{
    return (float)rand() / (float)RAND_MAX;
}

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Shared Memory Per Block : " << prop.sharedMemPerBlock << endl;
    cout << "Threads Per Block : " << prop.maxThreadsPerBlock << endl;
    cout << "Regs Per Block : " << prop.regsPerBlock << endl;
    cout << "Multi Processor Count : " << prop.multiProcessorCount << endl;
    cout << "Max Threads : " << prop.maxThreadsDim[0] << endl;

    // set random seed
    srand(1);

    float *feature;
    float *interaction_weight;

    int *input_data;
    float *result;

    // set feature

    cudaMallocManaged(&feature, N_FEATURE * DIM_FEAT * sizeof(float));
    cudaMallocManaged(&interaction_weight, N_FIELD * N_FIELD * sizeof(float));

    // randomly set feature
    for (int i = 0; i < N_FEATURE; i++)
    {
        for (int j = 0; j < DIM_FEAT; j++)
        {
            feature[i * DIM_FEAT + j] = rand_float();
        }
    }

    // randomly set interaction weight
    for (int i = 0; i < N_FIELD; i++)
    {
        for (int j = i + 1; j < N_FIELD; j++)
        {
            interaction_weight[i * N_FIELD + j] = rand_float();
        }
    }

    // set input data

    cudaMallocManaged(&input_data, INPUT_CNT * N_FIELD * sizeof(int));

    // randomly set input data.
    for (int i = 0; i < INPUT_CNT * N_FIELD; i++)
    {
        input_data[i] = rand() % N_FEATURE;
    }

    cudaMallocManaged(&result, INPUT_CNT * sizeof(float));

    // run, measure time

    dim3 blocks(64);
    dim3 threads(32);
    float start = clock();
    for (int i = 0; i < 10000; i++)
    {
        FwFM<<<blocks, threads>>>(result, feature, input_data, interaction_weight);
        cudaDeviceSynchronize();
    }
    float end = clock();
    float time = (end - start) / CLOCKS_PER_SEC;

    printf("time: %f\n", time);

    // check result
    printf("Output:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("result %d, %f\n", i, result[i]);
    }
    printf("result %d, %f\n", INPUT_CNT - 1, result[INPUT_CNT - 1]);
    printf("result %d, %f\n", INPUT_CNT, result[INPUT_CNT]);
    printf("result %d, %f\n", 2 * INPUT_CNT - 1, result[2 * INPUT_CNT - 1]);

    cudaFree(feature);
    cudaFree(interaction_weight);
    cudaFree(input_data);
    cudaFree(result);
    return 0;
}