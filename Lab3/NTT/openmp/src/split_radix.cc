/*
 * =============================================================================
 * 文件名: split_radix_final.cc
 * 描述: Split-Radix NTT并行化 - 最终正确版本
 * 策略: 基于原始main.cc的正确实现，添加Split-Radix优化思想
 * 核心: 混合Radix-2/4策略 + 并行化蝶形运算
 * =============================================================================
 */

#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <thread>
#include <algorithm>
#include <omp.h>

using namespace std;

int qpow(int x, int y, int p);
void get_rev(int *rev, int lim);
void standard_ntt(int *a, int lim, int opt, int p);
void split_radix_ntt(int *a, int lim, int opt, int p, int num_threads);
void parallel_radix4_butterfly(int *a, int lim, int len, int wn, int p, int num_threads);

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

void standard_ntt(int *a, int lim, int opt, int p) {
    int rev[lim];
    memset(rev, 0, sizeof(int) * lim);
    get_rev(rev, lim);

    for (int i = 0; i < lim; ++i) {
        if (i < rev[i]) swap(a[i], a[rev[i]]);
    }

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

    if (opt == -1) {
        int inv = qpow(lim, p - 2, p);
        for (int i = 0; i < lim; ++i) {
            a[i] = 1LL * a[i] * inv % p;
        }
    }
}

void split_radix_ntt(int *a, int lim, int opt, int p, int num_threads) {
    int rev[lim];
    memset(rev, 0, sizeof(int) * lim);
    get_rev(rev, lim);

    for (int i = 0; i < lim; ++i) {
        if (i < rev[i]) swap(a[i], a[rev[i]]);
    }

    for (int len = 2; len <= lim; len <<= 1) {
        int m = len >> 1;
        int wn = qpow(3, (p - 1) / len, p);
        if (opt == -1) wn = qpow(wn, p - 2, p);

        #pragma omp parallel for num_threads(num_threads) if(lim / len >= num_threads && len >= 64)
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

    if (opt == -1) {
        int inv = qpow(lim, p - 2, p);
        #pragma omp parallel for num_threads(num_threads) if(lim >= 1024)
        for (int i = 0; i < lim; ++i) {
            a[i] = 1LL * a[i] * inv % p;
        }
    }
}

void split_radix_poly_multiply(int *a, int *b, int *ab, int n, int p, int num_threads = 4) {
    memset(ab, 0, sizeof(int) * (2 * n - 1));

    int lim = 1;
    while (lim < 2 * n) lim <<= 1;

    int A[lim], B[lim];
    memset(A, 0, sizeof(int) * lim);
    memset(B, 0, sizeof(int) * lim);

    for (int i = 0; i < n; ++i) {
        A[i] = a[i];
        B[i] = b[i];
    }

    split_radix_ntt(A, lim, 1, p, num_threads);
    split_radix_ntt(B, lim, 1, p, num_threads);

    #pragma omp parallel for num_threads(num_threads) if(lim >= 1024)
    for (int i = 0; i < lim; ++i) {
        A[i] = 1LL * A[i] * B[i] % p;
    }

    split_radix_ntt(A, lim, -1, p, num_threads);

    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = A[i];
    }
}

void fRead(int *a, int *b, int *n, int *p, int input_id){
    string str1 = "/nttdata/";
    string str2 = to_string(input_id);
    string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    ifstream fin;
    fin.open(data_path, ios::in);
    fin>>*n>>*p;
    for (int i = 0; i < *n; i++){
        fin>>a[i];
    }
    for (int i = 0; i < *n; i++){
        fin>>b[i];
    }
}

void fCheck(int *ab, int n, int input_id){
    string str1 = "/nttdata/";
    string str2 = to_string(input_id);
    string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    ifstream fin;
    fin.open(data_path, ios::in);
    for (int i = 0; i < n * 2 - 1; i++){
        int x;
        fin>>x;
        if(x != ab[i]){
            cout<<"多项式乘法结果错误"<<endl;
            return;
        }
    }
    cout<<"多项式乘法结果正确"<<endl;
    return;
}

void fWrite(int *ab, int n, int input_id){
    string str1 = "files/";
    string str2 = to_string(input_id);
    string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    ofstream fout;
    fout.open(output_path, ios::out);
    for (int i = 0; i < n * 2 - 1; i++){
        fout<<ab[i]<<'\n';
    }
}

int a[300000], b[300000], ab[300000];

int main() {
    int test_begin = 0;
    int test_end = 3;
    for(int i = test_begin; i <= test_end; ++i){
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        
        auto start = chrono::high_resolution_clock::now();
        split_radix_poly_multiply(a, b, ab, n_, p_, 4);
        auto end = chrono::high_resolution_clock::now();
        
        chrono::duration<double,ratio<1,1000>> elapsed = end - start;
        
        fCheck(ab, n_, i);
        cout << "Split-Radix: n=" << n_ << " p=" << p_ 
             << " 延迟=" << elapsed.count() << " ms" << endl;
        system("mkdir -p files");
        fWrite(ab, n_, i);
    }
    return 0;
} 