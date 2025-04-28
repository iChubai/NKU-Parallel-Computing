#!/bin/bash

# NTT函数内部性能分析脚本

# 创建一个专门用于分析NTT内部性能的简化版本
cat > ntt_internal.cc << 'EOF'
#include <cstring>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <arm_neon.h>

// 从main.cc复制必要的函数
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

void get_rev(int *rev, int lim) {
    for (int i = 0; i < lim; ++i) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) ? (lim >> 1) : 0);
    }
}

// 串行NTT变换
void ntt_serial(int *a, int lim, int opt, int p) {
    // 位反转置换
    int rev[lim];
    memset(rev, 0, sizeof(int) * lim);
    get_rev(rev, lim);

    auto start_bit_reverse = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < lim; ++i) {
        if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    }
    auto end_bit_reverse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_bit_reverse = end_bit_reverse - start_bit_reverse;
    std::cout << "Serial NTT bit reverse time: " << elapsed_bit_reverse.count() * 1000 << " ms" << std::endl;

    auto start_butterfly = std::chrono::high_resolution_clock::now();
    for (int len = 2; len <= lim; len <<= 1) {
        int m = len >> 1;
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
    auto end_butterfly = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_butterfly = end_butterfly - start_butterfly;
    std::cout << "Serial NTT butterfly time: " << elapsed_butterfly.count() * 1000 << " ms" << std::endl;

    auto start_inv = std::chrono::high_resolution_clock::now();
    if (opt == -1) {
        int inv = qpow(lim, p - 2, p);
        for (int i = 0; i < lim; ++i) {
            a[i] = 1LL * a[i] * inv % p;
        }
    }
    auto end_inv = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_inv = end_inv - start_inv;
    std::cout << "Serial NTT inverse scaling time: " << elapsed_inv.count() * 1000 << " ms" << std::endl;
}

// 从main_simd.cc复制必要的函数和类
typedef uint32_t u32;
typedef uint64_t u64;

// 辅助函数：计算模逆元
u32 inverse(u32 a, u32 m) {
    u32 u = 0, v = 1;
    u32 m0 = m;
    while (a != 0) {
        u32 q = m / a;
        u32 r = m % a;
        u32 t = u - q * v;
        m = a;
        a = r;
        u = v;
        v = t;
    }
    if (u < 0) u += m0;
    return u;
}

// Montgomery规约的实现
class MontgomeryReducer {
private:
    u32 mod;       // 模数
    u32 mod_inv;   // -mod^(-1) mod 2^32
    u32 r2;        // (2^64) % mod

public:
    MontgomeryReducer(u32 mod) : mod(mod) {
        // 计算 -mod^(-1) mod 2^32
        mod_inv = inverse(mod, 1U << 31);
        // 计算 (2^64) % mod
        r2 = 0;
        u64 r = 1;
        for (int i = 0; i < 64; ++i) {
            r = (r << 1) % mod;
        }
        r2 = r;
    }

    // 单个数值的Montgomery规约
    u32 reduce(u64 x) {
        u32 q = (u32)x * mod_inv;
        u64 m = (u64)q * mod;
        u32 y = (x - m) >> 32;
        return x < m ? y + mod : y;
    }

    // 将数值转换到Montgomery域
    u32 to_montgomery(u32 x) {
        return reduce((u64)x * r2);
    }

    // 将数值从Montgomery域转换回普通域
    u32 from_montgomery(u32 x) {
        return reduce((u64)x);
    }

    // 在Montgomery域中进行乘法
    u32 mul(u32 a, u32 b) {
        return reduce((u64)a * b);
    }
};

// SIMD优化的NTT变换
void ntt_simd(int *a, int lim, int opt, int p) {
    // 创建Montgomery规约器
    MontgomeryReducer reducer(p);

    // 位反转置换
    int rev[lim];
    memset(rev, 0, sizeof(int) * lim);
    get_rev(rev, lim);

    auto start_bit_reverse = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < lim; ++i) {
        if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    }
    auto end_bit_reverse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_bit_reverse = end_bit_reverse - start_bit_reverse;
    std::cout << "SIMD NTT bit reverse time: " << elapsed_bit_reverse.count() * 1000 << " ms" << std::endl;

    auto start_to_mont = std::chrono::high_resolution_clock::now();
    // 将所有数转换到Montgomery域
    for (int i = 0; i < lim; ++i) {
        a[i] = reducer.to_montgomery(a[i]);
    }
    auto end_to_mont = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_to_mont = end_to_mont - start_to_mont;
    std::cout << "SIMD NTT to Montgomery time: " << elapsed_to_mont.count() * 1000 << " ms" << std::endl;

    auto start_butterfly = std::chrono::high_resolution_clock::now();
    // 蝴蝶操作
    for (int len = 2; len <= lim; len <<= 1) {
        int m = len >> 1;
        int wn = qpow(3, (p - 1) / len, p);
        if (opt == -1) wn = qpow(wn, p - 2, p);

        // 将wn转换到Montgomery域
        wn = reducer.to_montgomery(wn);
    
        for (int i = 0; i < lim; i += len) {
            int w = reducer.to_montgomery(1);  // 在Montgomery域中的1
            for (int j = 0; j < m; ++j) {
                int u = a[i + j];
                int v = reducer.mul(a[i + j + m], w);
                a[i + j] = (u + v) % p;
                a[i + j + m] = (u - v + p) % p;
                w = reducer.mul(w, wn);
            }
        }
    }
    auto end_butterfly = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_butterfly = end_butterfly - start_butterfly;
    std::cout << "SIMD NTT butterfly time: " << elapsed_butterfly.count() * 1000 << " ms" << std::endl;

    auto start_inv = std::chrono::high_resolution_clock::now();
    // 如果是逆变换，需要除以lim
    if (opt == -1) {
        int inv = qpow(lim, p - 2, p);
        inv = reducer.to_montgomery(inv);
        for (int i = 0; i < lim; ++i) {
            a[i] = reducer.mul(a[i], inv);
        }
    }
    auto end_inv = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_inv = end_inv - start_inv;
    std::cout << "SIMD NTT inverse scaling time: " << elapsed_inv.count() * 1000 << " ms" << std::endl;

    auto start_from_mont = std::chrono::high_resolution_clock::now();
    // 将结果从Montgomery域转换回普通域
    for (int i = 0; i < lim; ++i) {
        a[i] = reducer.from_montgomery(a[i]);
    }
    auto end_from_mont = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_from_mont = end_from_mont - start_from_mont;
    std::cout << "SIMD NTT from Montgomery time: " << elapsed_from_mont.count() * 1000 << " ms" << std::endl;
}

// 测试函数
void test_ntt_internal(int size, int p) {
    // 创建测试数组
    int lim = 1;
    while (lim < size) lim <<= 1;
    
    int *a_serial = new int[lim];
    int *a_simd = new int[lim];
    
    // 初始化数组
    for (int i = 0; i < lim; ++i) {
        a_serial[i] = a_simd[i] = (i < size) ? (rand() % p) : 0;
    }
    
    std::cout << "Size: " << size << ", Modulus: " << p << std::endl;
    
    // 测量串行NTT性能
    auto start_serial = std::chrono::high_resolution_clock::now();
    ntt_serial(a_serial, lim, 1, p);
    auto end_serial = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_serial = end_serial - start_serial;
    std::cout << "Total Serial NTT time: " << elapsed_serial.count() * 1000 << " ms" << std::endl;
    
    // 测量SIMD NTT性能
    auto start_simd = std::chrono::high_resolution_clock::now();
    ntt_simd(a_simd, lim, 1, p);
    auto end_simd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_simd = end_simd - start_simd;
    std::cout << "Total SIMD NTT time: " << elapsed_simd.count() * 1000 << " ms" << std::endl;
    
    std::cout << "Speedup: " << elapsed_serial.count() / elapsed_simd.count() << "x" << std::endl;
    std::cout << std::endl;
    
    delete[] a_serial;
    delete[] a_simd;
}

int main() {
    // 测试不同大小和模数
    int sizes[] = {131072};  // 只测试最大规模
    int moduli[] = {7340033, 104857601, 469762049};
    
    for (int size : sizes) {
        for (int p : moduli) {
            test_ntt_internal(size, p);
        }
    }
    
    return 0;
}
EOF

# 编译性能测试程序
echo "编译NTT内部性能测试程序..."
g++ -O3 -march=native -o ntt_internal ntt_internal.cc -fopenmp

# 运行性能测试
echo "=== NTT函数内部性能测试 ==="
./ntt_internal

# 使用perf进行详细分析
echo "=== NTT函数内部热点分析 ==="
perf record -g -o perf_results/ntt_internal.data ./ntt_internal
perf report -i perf_results/ntt_internal.data | head -30

echo "NTT内部性能测试完成！"
