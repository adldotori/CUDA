// COMPILE : nvcc -o main.out main.cu  -I /usr/include/opencv2 -L/usr/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

#include "opencv2/core/cuda.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

__global__ void changeRGBToGrey(unsigned char *ori, unsigned char *trans, int width, int height)
{

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width)
    {
        unsigned char r = ori[(row * width + col) * 3];
        unsigned char g = ori[(row * width + col) * 3 + 1];
        unsigned char b = ori[(row * width + col) * 3 + 2];
        trans[row * width + col] = 0.21 * r + 0.71 * g + 0.07 * b;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " NAME" << endl;
        return 1;
    }

    Mat src_host = imread(argv[1], IMREAD_COLOR);

    int h = src_host.cols;
    int w = src_host.rows;
    int c = src_host.channels();

    cout << "Dimension: " << c << " x " << w << " x " << h << endl;

    unsigned char *ori, *trans;
    unsigned char *d_ori, *d_trans;

    // Allocate host memory
    ori = (unsigned char *)malloc(sizeof(unsigned char) * c * w * h);
    trans = (unsigned char *)malloc(sizeof(unsigned char) * w * h);

    // Allocate device memory
    cudaMalloc((void **)&d_ori, sizeof(unsigned char) * c * w * h);
    cudaMalloc((void **)&d_trans, sizeof(unsigned char) * w * h);

    // Copy image from Mat to unsigned char
    memcpy(ori, src_host.data, sizeof(unsigned char) * c * w * h);

    // Transfer data from host to device memory
    cudaMemcpy(d_ori, ori, sizeof(unsigned char) * c * w * h, cudaMemcpyHostToDevice);

    // Execute Kernel
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(ceil(w / (double)dimBlock.x), ceil(h / (double)dimBlock.y), 1);
    changeRGBToGrey<<<dimGrid, dimBlock>>>(d_ori, d_trans, w, h);

    // Transfer data back to host memory
    cudaMemcpy(trans, d_trans, sizeof(unsigned char) * w * h, cudaMemcpyDeviceToHost);

    // Copy image from unsigned char to Mat
    Mat result(h, w, CV_8UC1, trans);
    imwrite("grey.png", result);

    // Deallocate device memory
    cudaFree(d_ori);
    cudaFree(d_trans);

    // Deallocate host memory
    free(ori);
    free(trans);
}
