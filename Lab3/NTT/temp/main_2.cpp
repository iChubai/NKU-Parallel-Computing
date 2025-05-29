#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstdio> // For printf in kernel if needed, and for error messages

// 编译文件
// hipcc main_2.cpp -o mlp_forward
// 执行文件
// ./mlp_forward 或者 hipprof ./mlp_forward

#define HIP_CHECK(error) \
    do { \
        hipError_t err = error; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error in %s at line %d: %s\\n", \
                    __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


#define BATCH 1024
#define I 10
#define H 20
#define O 5
#define TILE_DIM 16 // Tile dimension for matmul kernel

// 主要修改函数

/**
 * @brief 执行矩阵乘法 C = A * B.
 *
 * 使用分块（Tiling）策略通过共享内存优化访存。
 * 每个线程块计算输出矩阵C的一个TILE_DIM x TILE_DIM的子块。
 *
 * @param A 输入矩阵A (M x K)
 * @param B 输入矩阵B (K x N)
 * @param C 输出矩阵C (M x N)
 * @param M 矩阵A的行数，矩阵C的行数
 * @param N 矩阵B的列数，矩阵C的列数
 * @param K 矩阵A的列数，矩阵B的行数
 */
__global__ void matmul_kernel(const double* A, const double* B, double* C, int M, int N, int K) {
    __shared__ double sA[TILE_DIM][TILE_DIM];
    __shared__ double sB[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    double sum = 0.0;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load A tile
        if (row < M && (t * TILE_DIM + tx) < K) {
            sA[ty][tx] = A[row * K + (t * TILE_DIM + tx)];
        } else {
            sA[ty][tx] = 0.0;
        }

        // Load B tile
        if (col < N && (t * TILE_DIM + ty) < K) {
            sB[ty][tx] = B[(t * TILE_DIM + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0;
        }
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * @brief 将偏置向量B广播加到矩阵A的每一行。
 *
 * A = A + B (B被广播)
 *
 * @param A 输入输出矩阵 (M x N)
 * @param B 偏置向量 (1 x N or N)
 * @param M 矩阵A的行数
 * @param N 矩阵A的列数 (也是偏置向量B的长度)
 */
__global__ void add_bias_broadcast_kernel(double* A, const double* B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        A[row * N + col] += B[col];
    }
}

/**
 * @brief 对矩阵A的每个元素应用ReLU激活函数.
 *
 * A[i] = max(0, A[i])
 *
 * @param A 输入输出矩阵
 * @param size 矩阵中元素的总数量
 */
__global__ void relu_kernel(double* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmax(0.0, A[idx]);
    }
}

void random_init(std::vector<double>& mat) {
    for (auto& val : mat) {
        val = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0; // 修正范围为 [-1, 1]
    }
}

int main() {
    std::cout << "Initializing data on host..." << std::endl;
    std::vector<double> h_X(BATCH * I), h_W1(I * H), h_B1(H), h_W2(H * O), h_B2(O);
    std::vector<double> h_H_before_relu(BATCH * H), h_H_activated(BATCH * H); // 中间结果
    std::vector<double> h_Y_before_bias(BATCH * O), h_Y(BATCH * O); // 最终输出

    srand(0); // For reproducible results
    random_init(h_X);
    random_init(h_W1);
    random_init(h_B1);
    random_init(h_W2);
    random_init(h_B2);

    std::cout << "Allocating memory on device..." << std::endl;
    double *d_X, *d_W1, *d_B1, *d_H_before_relu, *d_H_activated, *d_W2, *d_B2, *d_Y_before_bias, *d_Y;

    HIP_CHECK(hipMalloc(&d_X, BATCH * I * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_W1, I * H * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B1, H * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_H_before_relu, BATCH * H * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_H_activated, BATCH * H * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_W2, H * O * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B2, O * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_Y_before_bias, BATCH * O * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_Y, BATCH * O * sizeof(double)));

    std::cout << "Copying data from host to device..." << std::endl;
    HIP_CHECK(hipMemcpy(d_X, h_X.data(), BATCH * I * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_W1, h_W1.data(), I * H * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B1, h_B1.data(), H * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_W2, h_W2.data(), H * O * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B2, h_B2.data(), O * sizeof(double), hipMemcpyHostToDevice));

    // 定义线程块和网格维度
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 threadsPerBlock_relu(256); // For 1D kernels like ReLU

    std::cout << "Launching kernels..." << std::endl;

    // Hidden layer: H_before_relu = X * W1
    // C(M,N) = A(M,K) * B(K,N)
    // M=BATCH, K=I, N=H
    dim3 numBlocks1((H + TILE_DIM - 1) / TILE_DIM, (BATCH + TILE_DIM - 1) / TILE_DIM);
    hipLaunchKernelGGL(matmul_kernel, numBlocks1, threadsPerBlock, 0, 0, d_X, d_W1, d_H_before_relu, BATCH, H, I);
    HIP_CHECK(hipGetLastError());

    // Add bias B1: H_before_relu = H_before_relu + B1
    // M=BATCH, N=H
    dim3 numBlocks_bias1((H + threadsPerBlock.x -1) / threadsPerBlock.x, (BATCH + threadsPerBlock.y -1) / threadsPerBlock.y);
    hipLaunchKernelGGL(add_bias_broadcast_kernel, numBlocks_bias1, threadsPerBlock, 0, 0, d_H_before_relu, d_B1, BATCH, H);
    HIP_CHECK(hipGetLastError());
    
    // Apply ReLU: H_activated = ReLU(H_before_relu)
    dim3 numBlocks_relu_h((BATCH * H + threadsPerBlock_relu.x - 1) / threadsPerBlock_relu.x);
    hipLaunchKernelGGL(relu_kernel, numBlocks_relu_h, threadsPerBlock_relu, 0, 0, d_H_before_relu, BATCH * H); // In-place ReLU on d_H_before_relu
    HIP_CHECK(hipMemcpy(d_H_activated, d_H_before_relu, BATCH * H * sizeof(double), hipMemcpyDeviceToDevice)); // Copy to d_H_activated if not in-place
    HIP_CHECK(hipGetLastError());


    // Output layer: Y_before_bias = H_activated * W2
    // C(M,N) = A(M,K) * B(K,N)
    // M=BATCH, K=H, N=O
    dim3 numBlocks2((O + TILE_DIM - 1) / TILE_DIM, (BATCH + TILE_DIM - 1) / TILE_DIM);
    hipLaunchKernelGGL(matmul_kernel, numBlocks2, threadsPerBlock, 0, 0, d_H_activated, d_W2, d_Y_before_bias, BATCH, O, H);
    HIP_CHECK(hipGetLastError());

    // Add output bias B2: Y = Y_before_bias + B2
    // M=BATCH, N=O
    dim3 numBlocks_bias2((O + threadsPerBlock.x -1) / threadsPerBlock.x, (BATCH + threadsPerBlock.y -1) / threadsPerBlock.y);
    hipLaunchKernelGGL(add_bias_broadcast_kernel, numBlocks_bias2, threadsPerBlock, 0, 0, d_Y_before_bias, d_B2, BATCH, O);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(d_Y, d_Y_before_bias, BATCH*O*sizeof(double), hipMemcpyDeviceToDevice)); // Copy to final d_Y


    HIP_CHECK(hipDeviceSynchronize()); // 等待所有核函数执行完毕
    std::cout << "Kernels execution finished." << std::endl;

    std::cout << "Copying results from device to host..." << std::endl;
    HIP_CHECK(hipMemcpy(h_Y.data(), d_Y, BATCH * O * sizeof(double), hipMemcpyDeviceToHost));

    // Print a few output values for verification
    std::cout << "Output verification (first 5 batches, all outputs):" << std::endl;
    for (int i = 0; i < std::min(BATCH, 5); ++i) {
       std::cout << "Batch[" << i << "]: ";
       for (int j = 0; j < O; ++j)
           std::cout << h_Y[i * O + j] << " ";
       std::cout << std::endl;
    }

    std::cout << "Freeing device memory..." << std::endl;
    HIP_CHECK(hipFree(d_X));
    HIP_CHECK(hipFree(d_W1));
    HIP_CHECK(hipFree(d_B1));
    HIP_CHECK(hipFree(d_H_before_relu));
    HIP_CHECK(hipFree(d_H_activated));
    HIP_CHECK(hipFree(d_W2));
    HIP_CHECK(hipFree(d_B2));
    HIP_CHECK(hipFree(d_Y_before_bias));
    HIP_CHECK(hipFree(d_Y));

    std::cout << "MLP forward pass completed successfully." << std::endl;
    return 0;
}