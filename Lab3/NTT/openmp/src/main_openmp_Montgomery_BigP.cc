/*
 * ===========================================
 * 文件名: main_openmp_Montgomery_BigP.cc
 * 描述: 使用Montgomery规约实现OpenMP并行优化的NTT算法，支持大模数
 * 编译: g++ -O3 -march=native -fopenmp -std=c++17 main_openmp_Montgomery_BigP.cc -o ntt_openmp_montgomery_bigp
 * ===========================================
 */
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <omp.h>

// 类型定义
using u32 = uint32_t;
using u64 = uint64_t;
using i128 = __int128;
using u128 = unsigned __int128;

// 模数定义
const int MODS[3] = {7340033, 104857601, 469762049};  // 三个较小的模数，支持更大范围的CRT
const int PRIMITIVE_ROOT = 3;              // 原根
const u64 BIG_MOD = 7696582450348003ULL;   // 大模数

// Montgomery规约器基类
class MontgomeryReducer {
public:
    virtual ~MontgomeryReducer() = default;
    virtual u64 to_montgomery(u64 x) = 0;
    virtual u64 from_montgomery(u64 x) = 0;
    virtual u64 mul(u64 a, u64 b) = 0;
    virtual u64 mod() const = 0;
};

// 32位模数的Montgomery规约器
class MontgomeryReducer32 : public MontgomeryReducer {
private:
    u32 m_mod;       
    u32 m_inv;       // -mod^(-1) mod 2^32
    u32 r2;          // R^2 mod m, R = 2^32

public:
    MontgomeryReducer32(u32 mod) : m_mod(mod) {
        // 计算-mod^(-1) mod 2^32
        m_inv = 1;
        for (int i = 0; i < 5; ++i) {
            m_inv *= 2 - mod * m_inv;
        }
        m_inv = -m_inv;

        // 计算R^2 mod m，R = 2^32
        r2 = (1ULL << 32) % mod;
        r2 = (u64)r2 * r2 % mod;
    }

    // 单个数值的Montgomery规约
    u32 reduce32(u64 t) const {
        u32 m = ((u32)t * m_inv) & 0xFFFFFFFF;
        u128 sum_val = (u128)t + (u64)m * m_mod;
        u64 tr = sum_val >> 32;
        return (tr >= m_mod) ? tr - m_mod : tr;
    }

    // 实现虚函数
    u64 to_montgomery(u64 x) override {
        return reduce32((u64)(x % m_mod) * r2);
    }

    u64 from_montgomery(u64 x) override {
        return reduce32(x);
    }

    u64 mul(u64 a, u64 b) override {
        return reduce32((u64)a * b);
    }
    
    u64 mod() const override {
        return m_mod;
    }
};

// 64位模数的Montgomery规约器
class MontgomeryReducer64 : public MontgomeryReducer {
private:
    u64 m_mod;
    u64 m_inv;   // -mod^(-1) mod 2^64
    u64 r2;      // R^2 mod m, R = 2^64

public:
    MontgomeryReducer64(u64 mod) : m_mod(mod) {
        // 计算-mod^(-1) mod 2^64
        m_inv = 1;
        for (int i = 0; i < 6; ++i) {
            m_inv *= 2 - mod * m_inv;
        }
        m_inv = -m_inv;

        // 计算R^2 mod m，R = 2^64
        r2 = (1ULL << 32) % mod;
        r2 = (u64)r2 * r2 % mod;
        r2 = (u64)r2 * r2 % mod;  // 因为2^64 = (2^32)^2
    }

    // 单个数值的Montgomery规约
    u64 reduce64(u128 t) const {
        u64 m = ((u64)t * m_inv) & 0xFFFFFFFFFFFFFFFF;
        u128 tr = (t + (u128)m * m_mod) >> 64;
        return (tr >= m_mod) ? tr - m_mod : tr;
    }

    // 实现虚函数
    u64 to_montgomery(u64 x) override {
        return reduce64((u128)(x % m_mod) * r2);
    }

    u64 from_montgomery(u64 x) override {
        return reduce64(x);
    }

    u64 mul(u64 a, u64 b) override {
        return reduce64((u128)a * b);
    }
    
    u64 mod() const override {
        return m_mod;
    }
};

// 辅助函数：快速幂
u64 qpow(u64 x, u64 y, u64 p) {
    u64 res = 1;
    x %= p;
    while (y) {
        if (y & 1) res = (u128)res * x % p;
        x = (u128)x * x % p;
        y >>= 1;
    }
    return res;
}

// 辅助函数：计算逆元
u64 inverse(u64 a, u64 m) {
    return qpow(a, m - 2, m);
}

// 辅助函数：中国剩余定理合并两个结果
u64 crt2(u64 r1, u64 r2, u64 m1, u64 m2) {
    u64 m1_inv_m2 = inverse(m1 % m2, m2);
    u64 x1 = r1;
    u64 x2 = ((r2 + m2 - x1 % m2) * m1_inv_m2) % m2;
    return x1 + (u128)m1 * x2;
}

// 辅助函数：中国剩余定理合并三个结果
u64 crt3(u64 r1, u64 r2, u64 r3, u64 m1, u64 m2, u64 m3, u64 target_mod) {
    // 先合并前两个
    u64 r12 = crt2(r1, r2, m1, m2);
    u64 m12 = (u128)m1 * m2 % target_mod;
    
    // 再合并第三个
    u64 m12_inv_m3 = inverse(m12 % m3, m3);
    u64 x = ((r3 + m3 - r12 % m3) * m12_inv_m3) % m3;
    
    u64 result = (r12 + (u128)m12 * x) % target_mod;
    return result;
}

// 预计算位反转数组
void get_rev(int* rev_arr, int lim) {
    rev_arr[0] = 0; 
    for (int i = 1; i < lim; i++) {
        rev_arr[i] = (rev_arr[i >> 1] >> 1) | ((i & 1) ? (lim >> 1) : 0);
    }
}

/* -------- IO / check / write 函数 -------- */
void fRead(u64 *a, u64 *b, int *n, u64 *p, int id) {
    std::string path = "/nttdata/" + std::to_string(id) + ".in";
    std::ifstream fin(path);
    fin >> *n >> *p;
    for (int i = 0; i < *n; i++) fin >> a[i];
    for (int i = 0; i < *n; i++) fin >> b[i];
}

void fCheck(u64 *ab, int n, int id) {
    std::string path = "/nttdata/" + std::to_string(id) + ".out";
    std::ifstream fin(path);
    for (int i = 0; i < 2 * n - 1; i++) {
        u64 x;
        fin >> x;
        if (x != ab[i]) {
            std::cout << "结果错误 - 位置 " << i << ": 期望 " << x << ", 实际 " << ab[i] << std::endl;
            return;
        }
    }
    std::cout << "结果正确\n";
}

void fWrite(u64 *ab, int n, int input_id) {
    std::string str1 = "../files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++) {
        fout << ab[i] << '\n';
    }
}

/* ---------- 基于Montgomery规约的并行NTT ---------- */
template<typename Reducer>
void ntt_montgomery(u64 *a, int lim, int opt, Reducer &reducer) {
    u64 p = reducer.mod();
    
    // 位反转
    std::vector<int> revv(lim);
    get_rev(revv.data(), lim);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < lim; ++i)
        if (i < revv[i]) std::swap(a[i], a[revv[i]]);

    // 将所有数转换到Montgomery域
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < lim; ++i)
        a[i] = reducer.to_montgomery(a[i]);

    // NTT蝴蝶操作
    for (int len = 2; len <= lim; len <<= 1) {
        int m = len >> 1;
        u64 wn = qpow(PRIMITIVE_ROOT, (p - 1) / len, p);
        if (opt == -1) wn = qpow(wn, p - 2, p);
        
        // 将单位根转换到Montgomery域
        wn = reducer.to_montgomery(wn);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < lim; i += len) {
            u64 w = reducer.to_montgomery(1);  // Montgomery域中的1
            for (int j = 0; j < m; ++j) {
                u64 u = a[i + j];
                u64 v = reducer.mul(a[i + j + m], w);
                
                // 在Montgomery域中进行模加和模减
                u64 sum = u + v;
                if (sum >= p) sum -= p;
                
                u64 diff = u;
                if (diff < v) diff += p;
                diff -= v;
                
                a[i + j] = sum;
                a[i + j + m] = diff;
                
                // 更新旋转因子
                w = reducer.mul(w, wn);
            }
        }
    }

    // 如果是逆变换，需要除以lim
    if (opt == -1) {
        u64 inv = qpow(lim, p - 2, p);
        inv = reducer.to_montgomery(inv);
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < lim; ++i)
            a[i] = reducer.mul(a[i], inv);
    }

    // 将结果从Montgomery域转换回普通域
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < lim; ++i)
        a[i] = reducer.from_montgomery(a[i]);
}

/* ---------- 处理32位模数的多项式乘法 ---------- */
void poly_multiply_montgomery32(u64 *a, u64 *b, u64 *ab, int n, u32 p) {
    memset(ab, 0, sizeof(u64) * (2 * n - 1));
    int lim = 1; while (lim < 2 * n) lim <<= 1;
    
    std::vector<u64> A(lim, 0), B(lim, 0);
    for (int i = 0; i < n; ++i) {
        A[i] = a[i] % p;
        B[i] = b[i] % p;
    }
    
    // 创建Montgomery规约器
    MontgomeryReducer32 reducer(p);
    
    // 执行NTT
    ntt_montgomery(A.data(), lim, 1, reducer);
    ntt_montgomery(B.data(), lim, 1, reducer);
    
    // 点乘 - 使用Montgomery乘法
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < lim; ++i)
        A[i] = reducer.mul(A[i], B[i]);
    
    // 执行INTT
    ntt_montgomery(A.data(), lim, -1, reducer);
    
    // 复制结果
    for (int i = 0; i < 2 * n - 1; ++i)
        ab[i] = A[i];
}

/* ---------- 处理64位模数的多项式乘法 ---------- */
void poly_multiply_montgomery64(u64 *a, u64 *b, u64 *ab, int n, u64 p) {
    memset(ab, 0, sizeof(u64) * (2 * n - 1));
    int lim = 1; while (lim < 2 * n) lim <<= 1;
    
    std::vector<u64> A(lim, 0), B(lim, 0);
    for (int i = 0; i < n; ++i) {
        A[i] = a[i] % p;
        B[i] = b[i] % p;
    }
    
    // 创建Montgomery规约器
    MontgomeryReducer64 reducer(p);
    
    // 执行NTT
    ntt_montgomery(A.data(), lim, 1, reducer);
    ntt_montgomery(B.data(), lim, 1, reducer);
    
    // 点乘 - 使用Montgomery乘法
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < lim; ++i)
        A[i] = reducer.mul(A[i], B[i]);
    
    // 执行INTT
    ntt_montgomery(A.data(), lim, -1, reducer);
    
    // 复制结果
    for (int i = 0; i < 2 * n - 1; ++i)
        ab[i] = A[i];
}

/* ---------- 处理大模数的多项式乘法（使用CRT） ---------- */
void poly_multiply_crt(u64 *a, u64 *b, u64 *ab, int n, u64 mod) {
    int lim = 1; while (lim < 2 * n) lim <<= 1;
    
    // 为三个模数的结果分配空间
    std::vector<std::vector<u64>> results(3, std::vector<u64>(2 * n - 1, 0));
    
    // 并行计算三个模数下的NTT
    #pragma omp parallel for num_threads(3)
    for (int t = 0; t < 3; ++t) {
        u32 current_mod = MODS[t];
        std::vector<u64> A(lim, 0), B(lim, 0);
        
        for (int i = 0; i < n; ++i) {
            A[i] = a[i] % current_mod;
            B[i] = b[i] % current_mod;
        }
        
        // 创建Montgomery规约器
        MontgomeryReducer32 reducer(current_mod);
        
        // 执行NTT、点乘和INTT
        ntt_montgomery(A.data(), lim, 1, reducer);
        ntt_montgomery(B.data(), lim, 1, reducer);
        
        // 点乘 - 使用Montgomery乘法
        for (int i = 0; i < lim; ++i)
            A[i] = reducer.mul(A[i], B[i]);
        
        ntt_montgomery(A.data(), lim, -1, reducer);
        
        // 保存结果
        for (int i = 0; i < 2 * n - 1; ++i)
            results[t][i] = A[i];
    }
    
    // 使用CRT合并结果
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = crt3(results[0][i], results[1][i], results[2][i], 
                   MODS[0], MODS[1], MODS[2], mod);
    }
}

/* ---------- 主函数 ---------- */
u64 a[300000], b[300000], ab[600000];

int main() {
    // 测试范围
    const int test_begin = 0;
    const int test_end = 4;
    
    for (int id = test_begin; id <= test_end; ++id) {
        int n;
        u64 p;
        
        // 读取输入
        fRead(a, b, &n, &p, id);
        memset(ab, 0, sizeof(ab));
        
        // 计时并执行NTT乘法
        auto start = std::chrono::high_resolution_clock::now();
        
        if (p > UINT32_MAX) {
            // 对于超大模数，使用CRT
            poly_multiply_crt(a, b, ab, n, p);
        } else {
            // 处理32位模数
            poly_multiply_montgomery32(a, b, ab, n, (u32)p);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> elapsed = end - start;
        double execution_time = elapsed.count();
        
        // 验证和输出
        fCheck(ab, n, id);
        std::cout << "平均延迟 (n = " << n << ", p = " << p << "): " 
                 << execution_time << " ms" << std::endl;
        fWrite(ab, n, id);
    }
    
    return 0;
} 