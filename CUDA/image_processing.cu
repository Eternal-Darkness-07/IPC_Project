#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

#define MAX_KERNEL_SIZE 31

__constant__ float d_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

// 🔹 Generate Gaussian Kernel (CPU)
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

// 🔹 Gaussian Blur Kernel
__global__ void gaussianBlur(unsigned char* src, unsigned char* dst, int W, int H, int kSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int half = kSize / 2;

    if (x < half || y < half || x >= W-half || y >= H-half) return;

    float sum = 0;

    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int idx = (y + ky) * W + (x + kx);
            int kIdx = (ky + half) * kSize + (kx + half);
            sum += d_kernel[kIdx] * src[idx];
        }
    }

    sum = min(255.0f, max(0.0f, sum));
    dst[y * W + x] = (unsigned char)sum;
}

// 🔹 Sobel Kernel
__global__ void sobel(unsigned char* src, unsigned char* dst, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= W-1 || y >= H-1) return;

    int Gx[3][3] = {
        {-1,0,1},
        {-2,0,2},
        {-1,0,1}
    };

    int Gy[3][3] = {
        {-1,-2,-1},
        { 0, 0, 0},
        { 1, 2, 1}
    };

    int sumX = 0, sumY = 0;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int val = src[(y+ky)*W + (x+kx)];
            sumX += val * Gx[ky+1][kx+1];
            sumY += val * Gy[ky+1][kx+1];
        }
    }

    int mag = abs(sumX) + abs(sumY);
    mag = min(255, mag);

    dst[y * W + x] = mag;
}

int main() {

    // 🔹 Load image (COLOR → match CPU)
    Mat img_color = imread("../input.jpg", IMREAD_COLOR);

    if (img_color.empty()) {
        cout << "Image not found!\n";
        return -1;
    }

    // 🔹 Convert to grayscale (same pipeline)
    Mat img_gray;
    cvtColor(img_color, img_gray, COLOR_BGR2GRAY);

    int W = img_gray.cols;
    int H = img_gray.rows;

    // 🔥 Dynamic kernel params
    int kSize = 11;
    float sigma = 5.0;

    auto kernel2D = generateKernel(kSize, sigma);

    vector<float> h_kernel(MAX_KERNEL_SIZE * MAX_KERNEL_SIZE, 0);

    for (int i = 0; i < kSize; i++) {
        for (int j = 0; j < kSize; j++) {
            h_kernel[i * kSize + j] = kernel2D[i][j];
        }
    }

    cudaMemcpyToSymbol(d_kernel, h_kernel.data(),
                       MAX_KERNEL_SIZE * MAX_KERNEL_SIZE * sizeof(float));

    // 🔹 GPU memory
    unsigned char *d_src, *d_blur, *d_edge;
    cudaMalloc(&d_src, W * H);
    cudaMalloc(&d_blur, W * H);
    cudaMalloc(&d_edge, W * H);

    cudaMemcpy(d_src, img_gray.data, W * H, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((W + 15)/16, (H + 15)/16);

    // 🔥 CUDA TIMING (SEPARATE)
    cudaEvent_t start, blur_done, sobel_done, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&blur_done);
    cudaEventCreate(&sobel_done);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 🔹 Gaussian Blur
    gaussianBlur<<<grid, block>>>(d_src, d_blur, W, H, kSize);
    cudaEventRecord(blur_done);

    // 🔹 Sobel
    sobel<<<grid, block>>>(d_blur, d_edge, W, H);
    cudaEventRecord(sobel_done);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float blur_ms, sobel_ms, total_ms;

    cudaEventElapsedTime(&blur_ms, start, blur_done);
    cudaEventElapsedTime(&sobel_ms, blur_done, sobel_done);
    cudaEventElapsedTime(&total_ms, start, stop);

    cout << "CUDA Blur Time: " << blur_ms / 1000.0 << " sec\n";
    cout << "CUDA Sobel Time: " << sobel_ms / 1000.0 << " sec\n";
    cout << "CUDA Total Time: " << total_ms / 1000.0 << " sec\n";

    // 🔹 Copy back
    Mat blur(H, W, CV_8UC1);
    Mat edge(H, W, CV_8UC1);

    cudaMemcpy(blur.data, d_blur, W * H, cudaMemcpyDeviceToHost);
    cudaMemcpy(edge.data, d_edge, W * H, cudaMemcpyDeviceToHost);

    imwrite("output_cuda_blur.jpg", blur);
    imwrite("output_cuda_sobel.jpg", edge);

    cudaFree(d_src);
    cudaFree(d_blur);
    cudaFree(d_edge);

    return 0;
}