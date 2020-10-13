#include <cstdio>
#include<iostream>
#include<fstream>
#include<ctime>
#include <cuda_runtime.h>

#define TILE_WIDTH 8

using namespace std;

//kernel function used by im2col(executed on GPU)
//reshape the feature map such that convolution can be executed as matrix multiplication
__global__ void im2colkernel(float* M, float* new_M, int N, int C, int H, int W, int K)
{

    int H_ = H - K + 1;
    int W_ = W - K + 1;

    int bx = blockIdx.x;//the feature map number

    int tx = threadIdx.x;//the index of partition on x axis
    int ty = threadIdx.y;//the index of partition on y axis
    int tc = threadIdx.z;//the index of partition on **channal** axis

    int x_range = H_ / blockDim.x;//suppose exact division
    int y_range = W_ / blockDim.y;//suppose exact division
    int c_range = C / blockDim.z;//suppose exact division


    for (int x = 0; x < x_range; x++)
    {
        for (int y = 0; y < y_range; y++)
        {
            for (int c = 0; c < c_range; c++)
            {
                for (int i = 0; i < K; i++)
                {
                    for (int j = 0; j < K; j++)
                    {
                        new_M[bx * H_ * W_ * C * K * K + ((tx * x_range + x) * W_ + (ty * y_range + y)) * C * K * K + (tc * c_range + c) * K * K + i * K + j] = M[bx * C * H * W + (tc * c_range + c) * H * W + (tx * x_range + x) * W + (ty * y_range + y) + i * W + j];
                    }
                }

            }
        }
    }
}

//wrapper function if im2colkernel
//accept a normal pointer
//return a cuda pointer
float* im2col(float* M, int N, int C, int H, int W, int K)
{
    float* Md, * new_M;

    int H_ = H - K + 1;
    int W_ = W - K + 1;

    int size_M = N * C * H * W * sizeof(float);
    int size_new_M = N * H_ * W_ * C * K * K * sizeof(float);

    //Transfer M  to device memory
    cudaMalloc((void**)&Md, size_M);
    cudaMemcpy(Md, M, size_M, cudaMemcpyHostToDevice);

    //allocate space for new_M
    cudaMalloc((void**)&new_M, size_new_M);

    dim3 dimBlock(6, 6, 8);//6 threads for with and height, and 8 threads for channal
    //Execute on GPU
    im2colkernel << <8, dimBlock >> > (Md, new_M, N, C, H, W, K);//8 blocks to handle different feature maps

    cudaFree(Md);


    return new_M;

}

//kernel function used by reshape_conv(executed on GPU)
//reshape the convolution kernel such that convolution can be executed as matrix multiplication
__global__ void reshape_convkernel(float* conv, float* new_conv, int F, int C, int K)
{
    int bx = blockIdx.x;//the number of convmap(f)

    int tx = threadIdx.x;//the number of channal(c)

    int b_range = F / gridDim.x;//suppose exact division

    int t_range = C / blockDim.x;//suppose exact division

    for (int f = 0; f < b_range; f++)
    {
        for (int c = 0; c < t_range; c++)
        {
            for (int i = 0; i < K; i++)
            {
                for (int j = 0; j < K; j++)
                {
                    new_conv[((c + t_range * tx) * K * K + i * K + j) * F + (f + b_range * bx)] = conv[(f + b_range * bx) * C * K * K + (c + t_range * tx) * K * K + i * K + j];
                }
            }
        }
    }
}

//wrapper function if reshape_convkernel
//accept and return a cuda pointer
float* reshape_conv(float* conv, int F, int C, int K)
{
    float* convd, * new_conv;

    int size_convd = K * K * C * F * sizeof(float);
    int size_new_conv = K * K * C * F * sizeof(float);

    //Transfer conv  to device memory
    cudaMalloc((void**)&convd, size_convd);
    cudaMemcpy(convd, conv, size_new_conv, cudaMemcpyHostToDevice);

    //allocate space for new_conv
    cudaMalloc((void**)&new_conv, size_new_conv);

    //Execute on GPU
    reshape_convkernel << <32, 32 >> > (convd, new_conv, F, C, K);// 32 blocks for F and 32 threads for channals

    cudaFree(convd);

    return new_conv;
}

//kernel function to reshape the output of matrix multiplication into the final result of convolution
//this will be invoked in MatrixMultiplication function
__global__ void reshape_outputkernel(float* P,float* ans,int size_P, int F, int N, int H_, int W_)
{
    int bx = blockIdx.x;//the number of feature map(N)
    int by = blockIdx.y;//the number for F

    int tx = threadIdx.x;//the number of height
    int ty = threadIdx.y;//the number of width

    int bx_range = N / gridDim.x;//suppose exact division
    int by_range = F / gridDim.y;//suppose exact division

    int tx_range = H_ / blockDim.x;//suppose exact division
    int ty_range = W_ / blockDim.y;//suppose exact division

    
    for (int n = 0; n < bx_range; n++)
    {
        for (int h_ = 0; h_ < tx_range; h_++)
        {
            for (int w_ = 0; w_ < ty_range; w_++)
            {
                for (int f = 0; f < by_range; f++)
                {
                    ans[(n+ bx_range*bx) * F * H_ * W_ + (f+by_range*by) * H_ * W_ + (h_+ tx_range*tx) * W_ + (w_+ ty_range*ty)] = P[(n + bx_range * bx) * F * H_ * W_ + (h_ + tx_range * tx) * W_ * F + (w_ + ty_range * ty) * F + (f + by_range * by)];
                }
            }
        }
    }
}

//kernel fuction to execute matrix multiplication(tiled) 
//this will be invoked in MatrixMultiplication function
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

    for (int i = 0; i < (k - 1) / TILE_WIDTH + 1; i++)//here i<(k - 1) / TILE_WIDTH + 1, this insures that every element in Pd will be calculated
    {
        if (Row < m && i * TILE_WIDTH + tx < k)//this if clause ensures no out-of-bounds in Md
            Mds[ty][tx] = Md[Row * k + (i * TILE_WIDTH + tx)];
        else
            Mds[ty][tx] = 0;
        if (i * TILE_WIDTH + ty < k && Col < n)//this if clause ensures no out-of-bounds in Nd
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
    if (Row < m && Col < n)//assign only when the index is legal
        Pd[Row * n + Col] = Pvalue;
}

//main function for matrix multi[lication.
//the result of M*M1 will be stored in P
//it will invoke Matrixkernel and reshape_outputkernel
void MatrixMultiplication(float* M, float* M1, float* P, int m, int k, int n,int F,int N,int H_,int W_)
{
    //M shape:m*k
    //N shape:k*n
    //P shape:m*n
    float* Pd;

    int size_P = m * n * sizeof(float);

    //Allocate P on the device
    cudaMalloc((void**)&Pd, size_P);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((n - 1) / TILE_WIDTH + 1, (m - 1) / TILE_WIDTH + 1);
    //Execute on GPU
    Matrixkernel << <dimGrid, dimBlock >> > (M, M1, Pd, m, k, n);

    float* ans;//hold the reshaped result of Pd
    cudaMalloc((void**)&ans, size_P);;

    dim3 dimBlock1(2, 8);
    dim3 dimGrid1(6, 6);

    reshape_outputkernel << <dimBlock1, dimGrid1 >> > (Pd,ans, size_P, F, N, H_, W_);

    //Transfer P from device to host
    cudaMemcpy(P, ans, size_P, cudaMemcpyDeviceToHost);
    cudaFree(M); cudaFree(M1); cudaFree(Pd); cudaFree(ans);//need to free M M1, they were allocated in other functions!

}

//7 loop version of convolution
//on CPU
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

//record the output tensor into log file
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
    //standard
    int N = 8;
    int C = 64;
    int H = 128;
    int W = 128;
    int F = 128;
    int K = 3;

    //test
    /*int N = 2;
    int C = 3;
    int H = 5;
    int W = 5;
    int F = 2;
    int K = 3;*/

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


    //new_M:point to cuda memory
    float* new_M = im2col(M, N, C, H, W, K);

    //new_kernel:point to cuda memory
    float* new_kernel = reshape_conv(kernel, F, C, K);

    //ans:normal pointer
    float* ans = new float[N * F * H_ * W_];

    MatrixMultiplication(new_M, new_kernel, ans, N * H_ * W_, K * K * C, F, F, N, H_, W_);

    endTime = clock();//计时结束

    cout << "Problem 2: The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    cout << "now writing ans into log..." << endl;
    log_output(ans, N, F, H_, W_);

    //delete the allocated space!
    delete M;
    delete kernel;
    delete ans;
    cout << "finish";
    
    return 0;
}