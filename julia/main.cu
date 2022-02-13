// COMPILE : nvcc -o main.out main.cu  -I /usr/include/opencv2 -L/usr/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#define MANDELBROT_ITERATIONS 256
#define JULIA_ITERATIONS 256

__global__ void julia(unsigned char *out, int width, int height)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < height && Col < width)
    {
        float x = Col / (float)width * 4.0 - 2.0;
        float y = Row / (float)height * 4.0 - 2.0;
        float x0 = x;
        float y0 = y;
        float c_real = -0.8;
        float c_imag = 0.156;
        int iteration = 0;
        while (x * x + y * y < 4 && iteration < JULIA_ITERATIONS)
        {
            float xtemp = x * x - y * y + c_real;
            y = 2 * x * y + c_imag;
            x = xtemp;
            iteration++;
        }
        out[(Row * width + Col) * 3] = 128;
        out[(Row * width + Col) * 3 + 1] = iteration;
        out[(Row * width + Col) * 3 + 2] = iteration;
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
    int width = 512, height = 512, channels = 3, time = 1;
    int N = time * channels * width * height;
    unsigned char *h_resultImg;
    unsigned char *d_resultImg;

    h_resultImg = (unsigned char *)malloc(N * sizeof(unsigned char));
    cudaMalloc((void **)&d_resultImg, N * sizeof(unsigned char));

    // Launch the Kernel
    const int block_size = 16;
    dim3 threads(block_size, block_size, 1);
    dim3 grid(ceil(width / (double)threads.x), ceil(height / (double)threads.y), 1);
    julia<<<grid, threads>>>(d_resultImg, width, height);

    // Copy the device result in device memory to the host result in host memory
    cudaMemcpy(h_resultImg, d_resultImg, N * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat resultImg(height, width, CV_8UC3);
    memcpy(resultImg.data, h_resultImg, width * height * channels);

    // Free device global memory
    cudaFree(d_resultImg);

    // Free host memory
    free(h_resultImg);

    // save image to filename.jpg
    cv::imwrite(file_name, resultImg);

    return 0;
}