/*
 * ===========================================
 * 文件名: main_DIF_DIT_simd_omp.cc
 * 描述: 使用 DIF+DIT 实现 NTT，并在蝶形和逐点乘阶段通过 OpenMP SIMD hint 自动向量化
 * ===========================================
 */

#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <omp.h>    // OpenMP

typedef long long ll;

void fRead(int *a, int *b, int *n, int *p, int input_id) {
    std::string dir = "/nttdata/";
    std::string fn  = dir + std::to_string(input_id) + ".in";
    std::ifstream fin(fn);
    fin >> *n >> *p;
    for (int i = 0; i < *n; ++i) fin >> a[i];
    for (int i = 0; i < *n; ++i) fin >> b[i];
}

// 校验输出
void fCheck(int *ab, int n, int input_id) {
    std::string dir = "/nttdata/";
    std::string fn  = dir + std::to_string(input_id) + ".out";
    std::ifstream fin(fn);
    for (int i = 0; i < n * 2 - 1; ++i) {
        int x; fin >> x;
        if (x != ab[i]) {
            std::cout << "多项式乘法结果错误\n";
            return;
        }
    }
    std::cout << "多项式乘法结果正确\n";
}

// 写输出
void fWrite(int *ab, int n, int input_id) {
    std::string dir = "files/";
    std::string fn  = dir + std::to_string(input_id) + ".out";
    std::ofstream fout(fn);
    for (int i = 0; i < n * 2 - 1; ++i) {
        fout << ab[i] << '\n';
    }
}

// 快速幂取模
inline ll fpow(ll a, ll b, int P) {
    ll res = 1;
    a %= P;
    while (b) {
        if (b & 1) res = res * a % P;
        a = a * a % P;
        b >>= 1;
    }
    return res;
}

// 计算所有 w[t]
void calc_powg(int w[], int G, int P, int gen) {
    w[0] = 1;
    const int g = fpow(gen, (P - 1) / G, P);
    for (int t = 0; (1 << (t + 1)) < G; ++t) {
        ll f = fpow(g, G >> (t + 2), P);
        w[1 << t] = f;
        for (int x = 1 << t; x < (1 << (t + 1)); ++x) {
            w[x] = (ll)f * w[x - (1 << t)] % P;
        }
    }
}

void DIF(int f[], int l, int P, int w[]) {
    int lim = 1 << l;
    for (int len = lim; len > 1; len >>= 1) {
        int half = len >> 1;
        for (int st = 0, t = 0; st < lim; st += len, ++t) {
            int wt = w[t];
            #pragma omp simd
            for (int i = st; i < st + half; ++i) {
                ll g = f[i];
                ll h = (ll)f[i + half] * wt % P;
                f[i]             = (int)((g + h) % P);
                f[i + half]      = (int)((P + g - h) % P);
            }
        }
    }
}

void pointwise_mul(int A[], int B[], int lim, int P) {
    #pragma omp simd
    for (int i = 0; i < lim; ++i) {
        A[i] = (int)((ll)A[i] * B[i] % P);
    }
}

void DIT(int f[], int l, int P, int w[]) {
    int lim = 1 << l;
    for (int len = 2; len <= lim; len <<= 1) {
        for (int st = 0, t = 0; st < lim; st += len, ++t) {
            for (int i = st; i < st + (len >> 1); ++i) {
                ll g = f[i], h = f[i + (len >> 1)];
                f[i]            = (int)((g + h) % P);
                f[i + (len >> 1)] = (int)(((ll)(P + g - h) * w[t]) % P);
            }
        }
    }
    ll invl = fpow(lim, P - 2, P);
    for (int i = 0; i < lim; ++i) {
        f[i] = (int)(invl * f[i] % P);
    }
    std::reverse(f + 1, f + lim);
}

void poly_multiply_optimized(int *a, int *b, int *ab, int n, int p, int gen = 3) {
    memset(ab, 0, sizeof(int) * (2 * n - 1));
    int l = 0; while ((1 << l) < 2 * n) ++l;
    int lim = 1 << l;

    int *A = new int[lim]();
    int *B = new int[lim]();
    int *w = new int[lim]();

    std::copy(a, a + n, A);
    std::copy(b, b + n, B);

    calc_powg(w, lim, p, gen);

    DIF(A, l, p, w);
    DIF(B, l, p, w);

    pointwise_mul(A, B, lim, p);

    DIT(A, l, p, w);

    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = A[i];
    }

    delete[] A;
    delete[] B;
    delete[] w;
}

int a[300000], b[300000], ab[600000];

int main(int argc, char *argv[]) {
    int test_begin = 0, test_end = 3;
    for (int i = test_begin; i <= test_end; ++i) {
        int n_, p_;
        fRead(a, b, &n_, &p_, i);

        auto start = std::chrono::high_resolution_clock::now();
        poly_multiply_optimized(a, b, ab, n_, p_);
        auto end   = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        fCheck(ab, n_, i);
        std::cout << "n=" << n_ << " p=" << p_
                  << " latency=" << elapsed.count() << " ms\n";
        fWrite(ab, n_, i);
    }
    return 0;
}
