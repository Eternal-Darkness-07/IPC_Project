#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

#define MAX_KERNEL_SIZE 31

__constant__ float d_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

// 🔹 Kernel generation (CPU)
vector<vector<float>> generateKernel(int kSize, float sigma) {
    vector<vector<float>> kernel(kSize, vector<float>(kSize));
    int half = kSize / 2;
    float sum = 0;

    for (int i = -half; i <= half; i++) {
        for (int j = -half; j <= half; j++) {
            float val = exp(-(i*i + j*j) / (2 * sigma * sigma));
            kernel[i + half][j + half] = val;
            sum += val;
        }
    }

    for (int i = 0; i < kSize; i++)
        for (int j = 0; j < kSize; j++)
            kernel[i][j] /= sum;

    return kernel;
}

// 🔥 COLOR BLUR
__global__ void gaussianBlurColor(uchar3* src, uchar3* dst, int W, int H, int kSize) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int half = kSize / 2;

    if (x < half || y < half || x >= W-half || y >= H-half) return;

    float b=0,g=0,r=0;

    for (int ky=-half; ky<=half; ky++) {
        for (int kx=-half; kx<=half; kx++) {

            int idx = (y+ky)*W + (x+kx);
            int kIdx = (ky+half)*kSize + (kx+half);

            uchar3 p = src[idx];
            float k = d_kernel[kIdx];

            b += k*p.x;
            g += k*p.y;
            r += k*p.z;
        }
    }

    uchar3 out;
    out.x = min(255.0f, max(0.0f, b));
    out.y = min(255.0f, max(0.0f, g));
    out.z = min(255.0f, max(0.0f, r));

    dst[y*W + x] = out;
}

// 🔥 RGB → GRAY (GPU)
__global__ void rgb2gray(uchar3* src, unsigned char* dst, int W, int H) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    uchar3 p = src[y*W + x];

    dst[y*W + x] = 0.299f*p.z + 0.587f*p.y + 0.114f*p.x;
}

// 🔥 SOBEL
__global__ void sobel(unsigned char* src, unsigned char* dst, int W, int H) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= W-1 || y >= H-1) return;

    int Gx[3][3] = {
        {-1,0,1},{-2,0,2},{-1,0,1}
    };

    int Gy[3][3] = {
        {-1,-2,-1},{0,0,0},{1,2,1}
    };

    int sx=0, sy=0;

    for (int ky=-1; ky<=1; ky++) {
        for (int kx=-1; kx<=1; kx++) {
            int val = src[(y+ky)*W + (x+kx)];
            sx += val * Gx[ky+1][kx+1];
            sy += val * Gy[ky+1][kx+1];
        }
    }

    int mag = abs(sx) + abs(sy);
    dst[y*W + x] = min(255, mag);
}

int main() {

    Mat img = imread("../input.jpg", IMREAD_COLOR);
    if (img.empty()) return -1;

    int W = img.cols, H = img.rows;

    // 🔹 Kernel
    int kSize = 11;
    float sigma = 5.0;

    auto kernel2D = generateKernel(kSize, sigma);

    vector<float> h_kernel(MAX_KERNEL_SIZE*MAX_KERNEL_SIZE,0);
    for(int i=0;i<kSize;i++)
        for(int j=0;j<kSize;j++)
            h_kernel[i*kSize+j]=kernel2D[i][j];

    cudaMemcpyToSymbol(d_kernel, h_kernel.data(),
        MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*sizeof(float));

    // 🔹 GPU memory
    uchar3 *d_src, *d_blur;
    unsigned char *d_gray, *d_edge;

    cudaMalloc(&d_src, W*H*sizeof(uchar3));
    cudaMalloc(&d_blur, W*H*sizeof(uchar3));
    cudaMalloc(&d_gray, W*H);
    cudaMalloc(&d_edge, W*H);

    cudaMemcpy(d_src, img.ptr<uchar3>(), W*H*sizeof(uchar3), cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((W+15)/16, (H+15)/16);

    // 🔥 TIMING
    cudaEvent_t start, blur_done, gray_done, sobel_done, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&blur_done);
    cudaEventCreate(&gray_done);
    cudaEventCreate(&sobel_done);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    gaussianBlurColor<<<grid,block>>>(d_src,d_blur,W,H,kSize);
    cudaEventRecord(blur_done);

    rgb2gray<<<grid,block>>>(d_blur,d_gray,W,H);
    cudaEventRecord(gray_done);

    sobel<<<grid,block>>>(d_gray,d_edge,W,H);
    cudaEventRecord(sobel_done);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float blur_ms, edge_ms, total_ms;

    cudaEventElapsedTime(&blur_ms, start, blur_done);
    cudaEventElapsedTime(&edge_ms, gray_done, sobel_done);
    cudaEventElapsedTime(&total_ms, start, stop);

    cout << "CUDA Blur Time: " << blur_ms/1000.0 << " sec\n";
    cout << "CUDA Edge Time: " << edge_ms/1000.0 << " sec\n";
    cout << "CUDA Total Time: " << total_ms/1000.0 << " sec\n";

    // 🔹 Copy results
    Mat blur(H,W,CV_8UC3);
    Mat edge(H,W,CV_8UC1);

    cudaMemcpy(blur.data, d_blur, W*H*sizeof(uchar3), cudaMemcpyDeviceToHost);
    cudaMemcpy(edge.data, d_edge, W*H, cudaMemcpyDeviceToHost);

    imwrite("output_cuda_blur.jpg", blur);
    imwrite("output_cuda_sobel.jpg", edge);

    return 0;
}