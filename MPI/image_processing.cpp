#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

using namespace std;
using namespace cv;

// 🔹 Kernel
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

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Mat img;
    int W, H;

    if (rank == 0) {
        img = imread("../input.jpg", IMREAD_COLOR);
        if (img.empty()) {
            cout << "Image not found\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        W = img.cols;
        H = img.rows;
    }

    // 🔹 Broadcast dimensions
    MPI_Bcast(&W, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&H, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_proc = H / size;

    Mat local_img(rows_per_proc, W, CV_8UC3);

    // 🔹 Scatter image
    MPI_Scatter(img.data, rows_per_proc * W * 3, MPI_UNSIGNED_CHAR,
                local_img.data, rows_per_proc * W * 3, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    // 🔹 Kernel
    int kSize = 11;
    float sigma = 5.0;
    auto kernel = generateKernel(kSize, sigma);
    int half = kSize / 2;

    Mat local_blur = local_img.clone();

    double start = MPI_Wtime();

    // 🔥 BLUR
    for (int y = half; y < rows_per_proc - half; y++) {
        for (int x = half; x < W - half; x++) {

            Vec3b pixel;

            for (int c = 0; c < 3; c++) {
                float sum = 0;

                for (int ky = -half; ky <= half; ky++) {
                    for (int kx = -half; kx <= half; kx++) {
                        sum += kernel[ky + half][kx + half] *
                               local_img.at<Vec3b>(y + ky, x + kx)[c];
                    }
                }

                pixel[c] = (uchar)sum;
            }

            local_blur.at<Vec3b>(y, x) = pixel;
        }
    }

    double mid = MPI_Wtime();

    // 🔥 GRAYSCALE
    Mat local_gray;
    cvtColor(local_blur, local_gray, COLOR_BGR2GRAY);

    Mat local_edge = Mat::zeros(local_gray.size(), CV_8UC1);

    int Gx[3][3] = {
        {-1,0,1},{-2,0,2},{-1,0,1}
    };

    int Gy[3][3] = {
        {-1,-2,-1},{0,0,0},{1,2,1}
    };

    // 🔥 SOBEL
    for (int y = 1; y < local_gray.rows - 1; y++) {
        for (int x = 1; x < W - 1; x++) {

            int sx=0, sy=0;

            for (int ky=-1; ky<=1; ky++) {
                for (int kx=-1; kx<=1; kx++) {
                    int val = local_gray.at<uchar>(y+ky, x+kx);
                    sx += val * Gx[ky+1][kx+1];
                    sy += val * Gy[ky+1][kx+1];
                }
            }

            int mag = abs(sx) + abs(sy);
            local_edge.at<uchar>(y,x) = min(255, mag);
        }
    }

    double end = MPI_Wtime();

    // 🔹 Final buffers
    Mat final_blur, final_edge;

    if (rank == 0) {
        final_blur = Mat(H, W, CV_8UC3);
        final_edge = Mat(H, W, CV_8UC1);
    }

    // 🔥 Gather BLUR
    MPI_Gather(local_blur.data, rows_per_proc * W * 3, MPI_UNSIGNED_CHAR,
               final_blur.data, rows_per_proc * W * 3, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    // 🔥 Gather EDGE
    MPI_Gather(local_edge.data, rows_per_proc * W, MPI_UNSIGNED_CHAR,
               final_edge.data, rows_per_proc * W, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "MPI Blur Time: " << (mid - start) << " sec\n";
        cout << "MPI Sobel Time: " << (end - mid) << " sec\n";
        cout << "MPI Total Time: " << (end - start) << " sec\n";

        imwrite("output_mpi_blur.jpg", final_blur);
        imwrite("output_mpi_sobel.jpg", final_edge);
    }

    MPI_Finalize();
    return 0;
}