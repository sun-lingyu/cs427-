#include <cstdio>
#include<iostream>
#include<fstream>
#include<ctime>
#include <cuda_runtime.h>

#define TILE_WIDTH 8

using namespace std;


float* im2col(float* M, int N, int C, int H, int W, int K)
{
    int kernel_num_h = H - K + 1;
    int kernel_num_w = W - K + 1;

    float* new_M = new float[N * kernel_num_h * kernel_num_w * C * K * K];

    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int h = 0; h < kernel_num_h; h++)
            {
                for (int w = 0; w < kernel_num_w; w++)
                {
                    for (int i = 0; i < K; i++)
                    {
                        for (int j = 0; j < K; j++)
                        {
                            new_M[n * kernel_num_h * kernel_num_w * C * K * K + (h * kernel_num_w + w) * C * K * K + c * K * K + i * K + j] = M[n * C * H * W + c * H * W + h * W + w + i * W + j];
                        }
                    }

                }
            }
        }
    }
    return new_M;
}
float* reshape_kernel(float* kernel, int F, int C, int K)
{
    float* new_kernel = new float[K * K * C * F];
    for (int f = 0; f < F; f++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int i = 0; i < K; i++)
            {
                for (int j = 0; j < K; j++)
                {
                    new_kernel[(c * K * K + i * K + j) * F + f] = kernel[f * C * K * K + c * K * K + i * K + j];
                }
            }
        }
    }
    return new_kernel;
}

float* reshape_output(float* M, int F, int N, int H_, int W_)
{
    float* ans = new float[N * F * H_ * W_];
    for (int n = 0; n < N; n++)
    {
        for (int h_ = 0; h_ < H_; h_++)
        {
            for (int w_ = 0; w_ < W_; w_++)
            {
                for (int f = 0; f < F; f++)
                {
                    ans[n * F * H_ * W_ + f * H_ * W_ + h_ * W_ + w_] = M[n * F * H_ * W_ + h_ * W_ * F + w_ * F + f];
                }
            }
        }
    }
    return ans;
}

__global__ void Matrixkernel(float* Md, float* Nd, float* Pd, int m, int k, int n)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for (int i = 0; i < (k - 1) / TILE_WIDTH + 1; i++)
    {
        if (Row < m && i * TILE_WIDTH + tx < k)
            Mds[ty][tx] = Md[Row * k + (i * TILE_WIDTH + tx)];
        else
            Mds[ty][tx] = 0;
        if (i * TILE_WIDTH + ty < k && Col < n)
            Nds[ty][tx] = Nd[Col + (i * TILE_WIDTH + ty) * n];
        else
            Nds[ty][tx] = 0;
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; j++)
        {
            Pvalue += Mds[ty][j] * Nds[j][tx];

        }
        __syncthreads();
    }
    if (Row < m && Col < n)
        Pd[Row * n + Col] = Pvalue;
}

void MatrixMultiplication(float* M, float* N, float* P, int m, int k, int n)
{
    //M shape:m*k
    //N shape:k*n
    //P shape:m*n
    float* Md, * Nd, * Pd;

    int size_M = m * k * sizeof(float);
    int size_N = k * n * sizeof(float);
    int size_P = m * n * sizeof(float);

    //Transfer M and N to device memory
    cudaMalloc((void**)&Md, size_M);
    cudaMemcpy(Md, M, size_M, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&Nd, size_N);
    cudaMemcpy(Nd, N, size_N, cudaMemcpyHostToDevice);

    //Allocate P on the device
    cudaMalloc((void**)&Pd, size_P);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((n - 1) / TILE_WIDTH + 1, (m - 1) / TILE_WIDTH + 1);
    //Execute on GPU
    Matrixkernel << <dimGrid, dimBlock >> > (Md, Nd, Pd, m, k, n);

    //Transfer P from device to host
    cudaMemcpy(P, Pd, size_P, cudaMemcpyDeviceToHost);
    cudaFree(Md); cudaFree(Nd); cudaFree(Pd);

}
void naiveconv(float* M, float* kernel, float* ans, int N, int C, int H, int W, int K, int F, int H_, int W_)
{
    for (int n = 0; n < N; n++)
    {
        for (int f = 0; f < F; f++)
        {
            for (int c = 0; c < C; c++)
            {
                for (int h = 0; h < H_; h++)
                {
                    for (int w = 0; w < W_; w++)
                    {
                        for (int i = 0; i < K; i++)
                        {
                            for (int j = 0; j < K; j++)
                            {
                                ans[n * F * H_ * W_ + f * H_ * W_ + h * W_ + w] += M[n * C * H * W + c * H * W + (h + i) * W + (w + j)] * kernel[f * C * K * K + c * K * K + i * K + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

void log_output(float* ans, int N, int F, int H_, int W_)
{
    ofstream out("./log");
    out << "the answer of the convolution is shown below:" << endl << endl;

    for (int i = 0; i < N; i++)
    {
        out << "img" << i << endl;
        for (int j = 0; j < F; j++)
        {
            out << "channel" << j << endl;
            for (int k = 0; k < H_; k++)
            {
                for (int h = 0; h < W_; h++)
                {
                    out << ans[i * F * H_ * W_ + j * H_ * W_ + k * W_ + h] << " ";
                }
                out << endl;
            }
            out << endl;
        }
        out << endl;
    }

    out.close();
}

int main()
{
    int N = 8;
    int C = 64;
    int H = 128;
    int W = 128;
    int F = 128;
    int K = 3;
    int H_ = H - K + 1;
    int W_ = W - K + 1;
    float* M = new float[N * C * H * W];
    for (int i = 0; i < N * C * H * W; i++)
    {
        M[i] = 2;
    }
    float* kernel = new float[F * C * K * K];
    for (int i = 0; i < F * C * K * K; i++)
    {
        kernel[i] = 2;
    }

    clock_t startTime, endTime;
    startTime = clock();//计时开始

    float* new_M = im2col(M, N, C, H, W, K);
    float* new_kernel = reshape_kernel(kernel, F, C, K);

    float* ans = new float[N * F * H_ * W_];

    MatrixMultiplication(new_M, new_kernel, ans, N * H_ * W_, K * K * C, F);

    /*for (int i = 0; i < N * H_ * W_; i++)
    {
        for (int j = 0; j < F; j++)
        {
            cout << ans[i * F + j];
        }
        cout << endl;
    }
    cout << endl;*/

    ans = reshape_output(ans, F, N, H_, W_);

    endTime = clock();//计时结束
    cout << "Problem 1: The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    cout << "now writing ans into log..." << endl;

    log_output(ans, N, F, H_, W_);

    cout << "finish";

    return 0;
}