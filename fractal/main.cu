// COMPILE : nvcc -o main.out main.cu  -I /usr/include/opencv2 -L/usr/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_shape -lopencv_videoio; ./main.out test.avi

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "opencv2/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#define MANDELBROT_ITERATIONS 256
#define JULIA_ITERATIONS 256

using namespace std;

// Input image has 3 channels corresponding to RGB
// The input image is encoded as unsigned characters [0, 255]
__global__ void fractal(unsigned char *out, int width, int height, int time)
{
    int time_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < height && Col < width && time_idx < time)
    {
        float x = Col / (float)width * 4.0 - 2.0;
        float y = Row / (float)height * 4.0 - 2.0;
        float ti = 2 * 3.141592 * time_idx / (float)time;
        float c_real = sin(ti) * 0.7 - 0.4;
        float c_imag = cos(ti) * 0.9;
        int iteration = 0;
        while (x * x + y * y < 4 && iteration < JULIA_ITERATIONS)
        {
            float xtemp = x * x - y * y + c_real;
            y = 2 * x * y + c_imag;
            x = xtemp;
            iteration++;
        }

        if (iteration == JULIA_ITERATIONS)
        {
            out[(time_idx * width * height + Row * width + Col) * 3] = 255;
            out[(time_idx * width * height + Row * width + Col) * 3 + 1] = 0;
            out[(time_idx * width * height + Row * width + Col) * 3 + 2] = 0;
        }
        else
        {
            out[(time_idx * width * height + Row * width + Col) * 3] = 0;
            out[(time_idx * width * height + Row * width + Col) * 3 + 1] = iteration;
            out[(time_idx * width * height + Row * width + Col) * 3 + 2] = 0;
        }
    }
}
void Usage(char prog_name[])
{
    fprintf(stderr, "Usage: %s <image output path>\n", prog_name);
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        Usage(argv[0]);
    }

    const char *file_name = argv[1];
    int width = 512, height = 512, channels = 3, time = 300;
    int N = time * channels * width * height;
    unsigned char *image;

    cudaMallocManaged((void **)&image, N * sizeof(unsigned char));

    // Launch the Kernel
    const int block_size = 16;
    dim3 threads(block_size, block_size, 1);
    dim3 grid(ceil(width / (double)threads.x), ceil(height / (double)threads.y), ceil(time / (double)threads.z));
    fractal<<<grid, threads>>>(image, width, height, time);

    // save to video
    cv::VideoWriter videoWriter;
    float videoFPS = 30.0f;
    videoWriter.open(file_name, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), videoFPS, cv::Size(width, height), true);
    //영상 저장 셋팅 실패시
    if (!videoWriter.isOpened())
    {
        std::cout << "Can't write video !!! check setting" << std::endl;
        return -1;
    }

    for (size_t k = 0; k < time; ++k)
    {
        cv::Mat resultImg(height, width, CV_8UC3);
        memcpy(resultImg.data, image + k * width * height * channels * sizeof(unsigned char), width * height * channels);
        // save image to filename.jpg
        // char name[50];
        // sprintf(name, "%d.png", k);
        // cv::imwrite(name, resultImg);
        videoWriter << resultImg;
    }

    cudaDeviceSynchronize();
    // Free device global memory
    cudaFree(image);

    return 0;
}