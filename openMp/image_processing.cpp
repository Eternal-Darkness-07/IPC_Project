#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace cv;

// Generate Gaussian Kernel
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

int main() {

    Mat img = imread("../input.jpg", IMREAD_COLOR);

    if (img.empty()) {
        cout << "Image not found!\n";
        return -1;
    }

    Mat out = img.clone();

    int H = img.rows, W = img.cols;

    int kSize = 11;
    float sigma = 5.0;

    auto kernel = generateKernel(kSize, sigma);
    int half = kSize / 2;

    auto start = chrono::high_resolution_clock::now();

    // 🔥 PARALLEL GAUSSIAN BLUR
    #pragma omp parallel for collapse(2)
    for (int y = half; y < H - half; y++) {
        for (int x = half; x < W - half; x++) {

            Vec3b pixel;

            for (int c = 0; c < 3; c++) {
                float sum = 0;

                for (int ky = -half; ky <= half; ky++) {
                    for (int kx = -half; kx <= half; kx++) {
                        sum += kernel[ky + half][kx + half] *
                               img.at<Vec3b>(y + ky, x + kx)[c];
                    }
                }

                pixel[c] = (uchar)sum;
            }

            out.at<Vec3b>(y, x) = pixel;
        }
    }

    auto mid = chrono::high_resolution_clock::now();

    // 🔥 SOBEL (convert to grayscale first)
    Mat gray;
    cvtColor(out, gray, COLOR_BGR2GRAY);

    Mat edges = Mat::zeros(gray.size(), CV_8UC1);

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1,-2,-1},
        { 0, 0, 0},
        { 1, 2, 1}
    };

    // 🔥 PARALLEL SOBEL
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < gray.rows - 1; y++) {
        for (int x = 1; x < gray.cols - 1; x++) {

            int sumX = 0;
            int sumY = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = gray.at<uchar>(y + ky, x + kx);

                    sumX += pixel * Gx[ky + 1][kx + 1];
                    sumY += pixel * Gy[ky + 1][kx + 1];
                }
            }

            int magnitude = abs(sumX) + abs(sumY);
            magnitude = min(255, magnitude);

            edges.at<uchar>(y, x) = magnitude;
        }
    }

    auto end = chrono::high_resolution_clock::now();

    // ⏱️ Timing
    cout << "OpenMP Blur Time: "
         << chrono::duration<double>(mid - start).count()
         << " sec\n";

    cout << "OpenMP Sobel Time: "
         << chrono::duration<double>(end - mid).count()
         << " sec\n";

    cout << "Total Time: "
         << chrono::duration<double>(end - start).count()
         << " sec\n";

    imwrite("output_omp_blur.jpg", out);
    imwrite("output_omp_sobel.jpg", edges);

    return 0;
}