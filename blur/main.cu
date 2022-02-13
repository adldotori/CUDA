// COMPILE : nvcc -o main.out main.cu  -I /usr/include/opencv2 -L/usr/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

#include "opencv2/core/cuda.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#define FILTER_SIZE 7

using namespace cv;
using namespace std;

__global__ void blur(unsigned char *ori, unsigned char *trans, int width, int height)
{

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width)
    {
        int R = 0;
        int G = 0;
        int B = 0;
        int cnt = 0;
        for (int i = -(FILTER_SIZE - 1) / 2; i <= (FILTER_SIZE - 1) / 2; i++)
        {
            for (int j = -(FILTER_SIZE - 1) / 2; j <= (FILTER_SIZE - 1) / 2; j++)
            {
                int r, g, b;
                int curRow = row + i;
                int curCol = col + j;
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width)
                {
                    cnt++;
                    r = ori[((row + i) * width + (col + j)) * 3];
                    g = ori[((row + i) * width + (col + j)) * 3 + 1];
                    b = ori[((row + i) * width + (col + j)) * 3 + 2];
                }
                R += r;
                G += g;
                B += b;
            }
        }
        trans[(row * width + col) * 3] = R / cnt;
        trans[(row * width + col) * 3 + 1] = G / cnt;
        trans[(row * width + col) * 3 + 2] = B / cnt;
    }
}

void getResource()
{
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    cudaDeviceProp dev_prop;
    for (int i = 0; i < dev_count; i++)
    {
        cudaGetDeviceProperties(&dev_prop, i);
        // decide if device has sufficient resources and capabilities
    }
    cout << "Device Count : " << dev_count << endl;
    cout << "Max Threads Per Block : " << dev_prop.maxThreadsPerBlock << endl;
    cout << "Multi Processor Count : " << dev_prop.multiProcessorCount << endl;
    cout << "Max Thread Dims : " << dev_prop.maxThreadsDim[0] << ", " << dev_prop.maxThreadsDim[1] << ", " << dev_prop.maxThreadsDim[2] << endl;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " NAME" << endl;
        return 1;
    }
    getResource();

    Mat src_host = imread(argv[1], IMREAD_COLOR);

    int h = src_host.cols;
    int w = src_host.rows;
    int c = src_host.channels();

    cout << "Dimension : " << c << " x " << w << " x " << h << endl;

    unsigned char *ori, *trans;
    unsigned char *d_ori, *d_trans;

    // Allocate host memory
    ori = (unsigned char *)malloc(sizeof(unsigned char) * c * w * h);
    trans = (unsigned char *)malloc(sizeof(unsigned char) * c * w * h);

    // Allocate device memory
    cudaMalloc((void **)&d_ori, sizeof(unsigned char) * c * w * h);
    cudaMalloc((void **)&d_trans, sizeof(unsigned char) * c * w * h);

    // Copy image from Mat to unsigned char
    memcpy(ori, src_host.data, sizeof(unsigned char) * c * w * h);

    // Transfer data from host to device memory
    cudaMemcpy(d_ori, ori, sizeof(unsigned char) * c * w * h, cudaMemcpyHostToDevice);

    // Execute Kernel
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(ceil(w / (double)dimBlock.x), ceil(h / (double)dimBlock.y), 1);
    blur<<<dimGrid, dimBlock>>>(d_ori, d_trans, w, h);

    // Transfer data back to host memory
    cudaMemcpy(trans, d_trans, sizeof(unsigned char) * c * w * h, cudaMemcpyDeviceToHost);

    // Copy image from unsigned char to Mat
    Mat result(h, w, CV_8UC3, trans);
    imwrite("blur.png", result);

    // Deallocate device memory
    cudaFree(d_ori);
    cudaFree(d_trans);

    // Deallocate host memory
    free(ori);
    free(trans);
}
