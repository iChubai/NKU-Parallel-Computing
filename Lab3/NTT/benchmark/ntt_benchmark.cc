#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <vector>
#include <random>
#include <algorithm>

// 快速幂函数，用于计算 x^y mod p
int qpow(int x, int y, int p) {
    int res = 1;
    x %= p;
    while (y) {
        if (y & 1) res = 1LL * res * x % p;
        x = 1LL * x * x % p;
        y >>= 1;
    }
    return res;
}

// 计算位反转表
void get_rev(int *rev, int lim) {
    for (int i = 0; i < lim; ++i) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) ? (lim >> 1) : 0);
    }
}

// NTT变换
void ntt(int *a, int lim, int opt, int p) {
    // 位反转置换
    int rev[lim];
    memset(rev, 0, sizeof(int) * lim);
    get_rev(rev, lim);

    for (int i = 0; i < lim; ++i) {
        if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    }

    // 蝴蝶操作
    for (int len = 2; len <= lim; len <<= 1) {
        int m = len >> 1;
        // 原根，对于不同的模数p，可能需要不同的原根
        int wn = qpow(3, (p - 1) / len, p);
        if (opt == -1) wn = qpow(wn, p - 2, p);

        for (int i = 0; i < lim; i += len) {
            int w = 1;
            for (int j = 0; j < m; ++j) {
                int u = a[i + j];
                int v = 1LL * a[i + j + m] * w % p;
                a[i + j] = (u + v) % p;
                a[i + j + m] = (u - v + p) % p;
                w = 1LL * w * wn % p;
            }
        }
    }

    // 如果是逆变换，需要除以lim
    if (opt == -1) {
        int inv = qpow(lim, p - 2, p);
        for (int i = 0; i < lim; ++i) {
            a[i] = 1LL * a[i] * inv % p;
        }
    }
}

// 多项式乘法函数
void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    // 清空结果数组
    memset(ab, 0, sizeof(int) * (2 * n - 1));

    // 计算NTT需要的长度（2的幂次）
    int lim = 1;
    while (lim < 2 * n) lim <<= 1;

    // 创建临时数组
    int A[lim], B[lim];
    memset(A, 0, sizeof(int) * lim);
    memset(B, 0, sizeof(int) * lim);

    // 复制输入数组到临时数组
    for (int i = 0; i < n; ++i) {
        A[i] = a[i];
        B[i] = b[i];
    }

    // 执行NTT变换
    ntt(A, lim, 1, p);
    ntt(B, lim, 1, p);

    // 点乘
    for (int i = 0; i < lim; ++i) {
        A[i] = 1LL * A[i] * B[i] % p;
    }

    // 执行逆NTT变换
    ntt(A, lim, -1, p);

    // 复制结果到输出数组
    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = A[i];
    }
}

// 分阶段测试NTT性能
void benchmark_ntt_stages(int n, int p, int runs = 5) {
    std::cout << "===== 测试NTT各阶段性能 (n=" << n << ", p=" << p << ") =====" << std::endl;
    
    // 生成随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, p - 1);
    
    int lim = 1;
    while (lim < 2 * n) lim <<= 1;
    
    int *a = new int[lim];
    int *b = new int[lim];
    int *c = new int[2 * n - 1];
    
    // 初始化数据
    for (int i = 0; i < n; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }
    for (int i = n; i < lim; ++i) {
        a[i] = 0;
        b[i] = 0;
    }
    
    // 测试各个阶段
    double time_bit_reverse = 0;
    double time_forward_ntt_a = 0;
    double time_forward_ntt_b = 0;
    double time_point_mul = 0;
    double time_inverse_ntt = 0;
    double time_total = 0;
    
    for (int run = 0; run < runs; ++run) {
        int A[lim], B[lim];
        memcpy(A, a, sizeof(int) * lim);
        memcpy(B, b, sizeof(int) * lim);
        
        // 总时间
        auto start_total = std::chrono::high_resolution_clock::now();
        
        // 1. 位反转置换 (A)
        auto start = std::chrono::high_resolution_clock::now();
        int rev[lim];
        memset(rev, 0, sizeof(int) * lim);
        get_rev(rev, lim);
        for (int i = 0; i < lim; ++i) {
            if (i < rev[i]) std::swap(A[i], A[rev[i]]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        time_bit_reverse += std::chrono::duration<double, std::ratio<1, 1000>>(end - start).count();
        
        // 2. 正向NTT (A)
        start = std::chrono::high_resolution_clock::now();
        // 蝴蝶操作
        for (int len = 2; len <= lim; len <<= 1) {
            int m = len >> 1;
            int wn = qpow(3, (p - 1) / len, p);
            
            for (int i = 0; i < lim; i += len) {
                int w = 1;
                for (int j = 0; j < m; ++j) {
                    int u = A[i + j];
                    int v = 1LL * A[i + j + m] * w % p;
                    A[i + j] = (u + v) % p;
                    A[i + j + m] = (u - v + p) % p;
                    w = 1LL * w * wn % p;
                }
            }
        }
        end = std::chrono::high_resolution_clock::now();
        time_forward_ntt_a += std::chrono::duration<double, std::ratio<1, 1000>>(end - start).count();
        
        // 3. 正向NTT (B)
        start = std::chrono::high_resolution_clock::now();
        // 位反转置换
        for (int i = 0; i < lim; ++i) {
            if (i < rev[i]) std::swap(B[i], B[rev[i]]);
        }
        // 蝴蝶操作
        for (int len = 2; len <= lim; len <<= 1) {
            int m = len >> 1;
            int wn = qpow(3, (p - 1) / len, p);
            
            for (int i = 0; i < lim; i += len) {
                int w = 1;
                for (int j = 0; j < m; ++j) {
                    int u = B[i + j];
                    int v = 1LL * B[i + j + m] * w % p;
                    B[i + j] = (u + v) % p;
                    B[i + j + m] = (u - v + p) % p;
                    w = 1LL * w * wn % p;
                }
            }
        }
        end = std::chrono::high_resolution_clock::now();
        time_forward_ntt_b += std::chrono::duration<double, std::ratio<1, 1000>>(end - start).count();
        
        // 4. 点乘
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < lim; ++i) {
            A[i] = 1LL * A[i] * B[i] % p;
        }
        end = std::chrono::high_resolution_clock::now();
        time_point_mul += std::chrono::duration<double, std::ratio<1, 1000>>(end - start).count();
        
        // 5. 逆NTT
        start = std::chrono::high_resolution_clock::now();
        // 位反转置换
        for (int i = 0; i < lim; ++i) {
            if (i < rev[i]) std::swap(A[i], A[rev[i]]);
        }
        // 蝴蝶操作
        for (int len = 2; len <= lim; len <<= 1) {
            int m = len >> 1;
            int wn = qpow(3, (p - 1) / len, p);
            wn = qpow(wn, p - 2, p);
            
            for (int i = 0; i < lim; i += len) {
                int w = 1;
                for (int j = 0; j < m; ++j) {
                    int u = A[i + j];
                    int v = 1LL * A[i + j + m] * w % p;
                    A[i + j] = (u + v) % p;
                    A[i + j + m] = (u - v + p) % p;
                    w = 1LL * w * wn % p;
                }
            }
        }
        // 除以lim
        int inv = qpow(lim, p - 2, p);
        for (int i = 0; i < lim; ++i) {
            A[i] = 1LL * A[i] * inv % p;
        }
        end = std::chrono::high_resolution_clock::now();
        time_inverse_ntt += std::chrono::duration<double, std::ratio<1, 1000>>(end - start).count();
        
        auto end_total = std::chrono::high_resolution_clock::now();
        time_total += std::chrono::duration<double, std::ratio<1, 1000>>(end_total - start_total).count();
    }
    
    // 计算平均时间
    time_bit_reverse /= runs;
    time_forward_ntt_a /= runs;
    time_forward_ntt_b /= runs;
    time_point_mul /= runs;
    time_inverse_ntt /= runs;
    time_total /= runs;
    
    // 输出结果
    std::cout << "位反转置换: " << time_bit_reverse << " us (" 
              << (time_bit_reverse / time_total * 100) << "%)" << std::endl;
    std::cout << "正向NTT (A): " << time_forward_ntt_a << " us (" 
              << (time_forward_ntt_a / time_total * 100) << "%)" << std::endl;
    std::cout << "正向NTT (B): " << time_forward_ntt_b << " us (" 
              << (time_forward_ntt_b / time_total * 100) << "%)" << std::endl;
    std::cout << "点乘: " << time_point_mul << " us (" 
              << (time_point_mul / time_total * 100) << "%)" << std::endl;
    std::cout << "逆NTT: " << time_inverse_ntt << " us (" 
              << (time_inverse_ntt / time_total * 100) << "%)" << std::endl;
    std::cout << "总时间: " << time_total << " us (100%)" << std::endl;
    
    delete[] a;
    delete[] b;
    delete[] c;
}

// 测试不同输入规模下的性能
void benchmark_input_size(int p, int runs = 5) {
    std::cout << "===== 测试不同输入规模下的性能 (p=" << p << ") =====" << std::endl;
    
    std::vector<int> sizes = {16, 64, 256, 1024, 4096, 16384, 65536, 131072};
    
    for (int n : sizes) {
        // 生成随机数据
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, p - 1);
        
        int *a = new int[n];
        int *b = new int[n];
        int *c = new int[2 * n - 1];
        
        // 初始化数据
        for (int i = 0; i < n; ++i) {
            a[i] = dis(gen);
            b[i] = dis(gen);
        }
        
        // 多次运行取平均值
        double total_time = 0;
        for (int run = 0; run < runs; ++run) {
            auto start = std::chrono::high_resolution_clock::now();
            poly_multiply(a, b, c, n, p);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::ratio<1, 1000>>(end - start).count();
        }
        
        double avg_time = total_time / runs;
        std::cout << "n = " << n << ": " << avg_time << " us" << std::endl;
        
        delete[] a;
        delete[] b;
        delete[] c;
    }
}

// 测试不同模数下的性能
void benchmark_modulus(int n, int runs = 5) {
    std::cout << "===== 测试不同模数下的性能 (n=" << n << ") =====" << std::endl;
    
    std::vector<int> moduli = {7340033, 104857601, 469762049};
    
    for (int p : moduli) {
        // 生成随机数据
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, p - 1);
        
        int *a = new int[n];
        int *b = new int[n];
        int *c = new int[2 * n - 1];
        
        // 初始化数据
        for (int i = 0; i < n; ++i) {
            a[i] = dis(gen);
            b[i] = dis(gen);
        }
        
        // 多次运行取平均值
        double total_time = 0;
        for (int run = 0; run < runs; ++run) {
            auto start = std::chrono::high_resolution_clock::now();
            poly_multiply(a, b, c, n, p);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::ratio<1, 1000>>(end - start).count();
        }
        
        double avg_time = total_time / runs;
        std::cout << "p = " << p << ": " << avg_time << " us" << std::endl;
        
        delete[] a;
        delete[] b;
        delete[] c;
    }
}

// 测试蝴蝶操作的性能
void benchmark_butterfly(int n, int p, int runs = 5) {
    std::cout << "===== 测试蝴蝶操作性能 (n=" << n << ", p=" << p << ") =====" << std::endl;
    
    // 生成随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, p - 1);
    
    int lim = 1;
    while (lim < 2 * n) lim <<= 1;
    
    int *a = new int[lim];
    
    // 初始化数据
    for (int i = 0; i < n; ++i) {
        a[i] = dis(gen);
    }
    for (int i = n; i < lim; ++i) {
        a[i] = 0;
    }
    
    // 位反转置换
    int rev[lim];
    memset(rev, 0, sizeof(int) * lim);
    get_rev(rev, lim);
    for (int i = 0; i < lim; ++i) {
        if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    }
    
    // 测试每一层蝴蝶操作的性能
    int layers = 0;
    for (int len = 2; len <= lim; len <<= 1) {
        layers++;
    }
    
    std::vector<double> layer_times(layers, 0);
    
    for (int run = 0; run < runs; ++run) {
        int A[lim];
        memcpy(A, a, sizeof(int) * lim);
        
        int layer = 0;
        for (int len = 2; len <= lim; len <<= 1) {
            auto start = std::chrono::high_resolution_clock::now();
            
            int m = len >> 1;
            int wn = qpow(3, (p - 1) / len, p);
            
            for (int i = 0; i < lim; i += len) {
                int w = 1;
                for (int j = 0; j < m; ++j) {
                    int u = A[i + j];
                    int v = 1LL * A[i + j + m] * w % p;
                    A[i + j] = (u + v) % p;
                    A[i + j + m] = (u - v + p) % p;
                    w = 1LL * w * wn % p;
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            layer_times[layer] += std::chrono::duration<double, std::ratio<1, 1000>>(end - start).count();
            layer++;
        }
    }
    
    // 计算平均时间
    for (int i = 0; i < layers; ++i) {
        layer_times[i] /= runs;
        std::cout << "第 " << (i + 1) << " 层蝴蝶操作: " << layer_times[i] << " us" << std::endl;
    }
    
    delete[] a;
}

// 测试取模操作的性能
void benchmark_modulo(int n, int p, int runs = 5) {
    std::cout << "===== 测试取模操作性能 (n=" << n << ", p=" << p << ") =====" << std::endl;
    
    // 生成随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, p - 1);
    
    int *a = new int[n];
    int *b = new int[n];
    
    // 初始化数据
    for (int i = 0; i < n; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }
    
    // 测试不同取模操作的性能
    double time_mod = 0;
    double time_if_mod = 0;
    
    for (int run = 0; run < runs; ++run) {
        // 1. 直接取模 (a * b) % p
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n; ++i) {
            int c = 1LL * a[i] * b[i] % p;
        }
        auto end = std::chrono::high_resolution_clock::now();
        time_mod += std::chrono::duration<double, std::ratio<1, 1000>>(end - start).count();
        
        // 2. 条件取模 (a + b >= p) ? (a + b - p) : (a + b)
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n; ++i) {
            int c = 1LL * a[i] * b[i];
            c = (c >= p) ? (c - p) : c;
            while (c >= p) c -= p;
            while (c < 0) c += p;
        }
        end = std::chrono::high_resolution_clock::now();
        time_if_mod += std::chrono::duration<double, std::ratio<1, 1000>>(end - start).count();
    }
    
    // 计算平均时间
    time_mod /= runs;
    time_if_mod /= runs;
    
    std::cout << "直接取模 (a * b) % p: " << time_mod << " us" << std::endl;
    std::cout << "条件取模 (条件判断): " << time_if_mod << " us" << std::endl;
    
    delete[] a;
    delete[] b;
}

int main() {
    int runs = 10; // 每个测试运行的次数
    
    // 1. 测试不同输入规模下的性能
    benchmark_input_size(7340033, runs);
    std::cout << std::endl;
    
    // 2. 测试不同模数下的性能
    benchmark_modulus(131072, runs);
    std::cout << std::endl;
    
    // 3. 测试NTT各阶段的性能
    benchmark_ntt_stages(131072, 7340033, runs);
    std::cout << std::endl;
    
    // 4. 测试蝴蝶操作的性能
    benchmark_butterfly(131072, 7340033, runs);
    std::cout << std::endl;
    
    // 5. 测试取模操作的性能
    benchmark_modulo(1000000, 7340033, runs);
    
    return 0;
}
