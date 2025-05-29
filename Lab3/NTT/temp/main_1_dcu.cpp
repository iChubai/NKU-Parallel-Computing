#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#define N 1024
#define M 2024
#define P 512

// 瓦块大小（建议 16、32 视硬件而定）
#define TS 16

// GPU 瓦块化矩阵乘法内核
__global__
void matmul_kernel(const double* __restrict__ A,
                   const double* __restrict__ B,
                   double* __restrict__ C,
                   int n, int m, int p) {
    // 共享内存存储 A 瓦块和 B 瓦块
    __shared__ double Asub[TS][TS];
    __shared__ double Bsub[TS][TS];

    // 全局行、列索引
    int row = blockIdx.y * TS + threadIdx.y;
    int col = blockIdx.x * TS + threadIdx.x;

    double sum = 0.0;

    // 对 k 维度分瓦块
    for (int t = 0; t < (m + TS - 1) / TS; ++t) {
        // 从全局拷贝到共享内存，越界则补 0
        int aRow = row;
        int aCol = t * TS + threadIdx.x;
        Asub[threadIdx.y][threadIdx.x] = 
            (aRow < n && aCol < m) ? A[aRow * m + aCol] : 0.0;

        int bRow = t * TS + threadIdx.y;
        int bCol = col;
        Bsub[threadIdx.y][threadIdx.x] = 
            (bRow < m && bCol < p) ? B[bRow * p + bCol] : 0.0;

        __syncthreads();

        // 在共享内存上做小瓦块的乘加
        #pragma unroll
        for (int k = 0; k < TS; ++k) {
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }
        __syncthreads();
    }

    // 写回全局内存
    if (row < n && col < p) {
        C[row * p + col] = sum;
    }
}

void init_matrix(std::vector<double>& mat) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& x : mat) x = dist(gen);
}

void matmul_cpu(const std::vector<double>& A,
                const std::vector<double>& B,
                std::vector<double>& C) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
}

bool validate(const std::vector<double>& ref,
              const std::vector<double>& test) {
    for (size_t i = 0; i < ref.size(); ++i)
        if (std::abs(ref[i] - test[i]) > 1e-6)
            return false;
    return true;
}

int main() {
    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P);
    init_matrix(A);
    init_matrix(B);
    matmul_cpu(A, B, C_ref);

    // 设备指针
    double *d_A, *d_B, *d_C;
    hipMalloc(&d_A, sizeof(double) * N * M);
    hipMalloc(&d_B, sizeof(double) * M * P);
    hipMalloc(&d_C, sizeof(double) * N * P);

    hipMemcpy(d_A, A.data(), sizeof(double) * N * M, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), sizeof(double) * M * P, hipMemcpyHostToDevice);
    hipMemset(d_C, 0, sizeof(double) * N * P);

    // 配置网格与线程块
    dim3 block(TS, TS);
    dim3 grid((P + TS - 1) / TS, (N + TS - 1) / TS);

    // 事件计时
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);

    // 启动内核
    hipLaunchKernelGGL(matmul_kernel, grid, block, 0, 0,
                       d_A, d_B, d_C, N, M, P);

    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float elapsed;
    hipEventElapsedTime(&elapsed, start, stop);
    double sec = elapsed / 1e3;

    hipMemcpy(C.data(), d_C, sizeof(double) * N * P, hipMemcpyDeviceToHost);

    // 验证结果
    bool ok = validate(C_ref, C);
    std::cout << "[HIP-Tiled] Valid: " << ok
              << "  Time: " << sec << " s"
              << "  GFLOPS: " << (2.0 * N * M * P / 1e9 / sec)
              << std::endl;

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return 0;
}
