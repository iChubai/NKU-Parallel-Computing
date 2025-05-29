/*
 * ===========================================
 * 文件名 main.cc
 * 描述: NTT 串行实现 + "朴素"多线程优化（并行前向 NTT）
 * ===========================================
 */
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <pthread.h>    
#include <vector>    

void fRead(int *a, int *b, int *n, int *p, int input_id){
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin(data_path);
    fin >> *n >> *p;
    for (int i = 0; i < *n; i++) fin >> a[i];
    for (int i = 0; i < *n; i++) fin >> b[i];
}

void fCheck(int *ab, int n, int input_id){
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin(data_path);
    for (int i = 0; i < n * 2 - 1; i++){
        int x; fin >> x;
        if(x != ab[i]){
            std::cout << "多项式乘法结果错误\n";
            return;
        }
    }
    std::cout << "多项式乘法结果正确\n";
}

void fWrite(int *ab, int n, int input_id){
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout(output_path);
    for (int i = 0; i < n * 2 - 1; i++){
        fout << ab[i] << '\n';
    }
}

int qpow(int x, int y, int p) {
    int res = 1;
    long long b = x % p;
    while (y) {
        if (y & 1) res = int(res * b % p);
        b = (b * b) % p;
        y >>= 1;
    }
    return res;
}

void get_rev(int *rev, int lim) {
    for (int i = 0; i < lim; ++i) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) ? (lim >> 1) : 0);
    }
}

void ntt(int *a, int lim, int opt, int p) {
    int rev[lim];
    get_rev(rev, lim);
    for (int i = 0; i < lim; ++i) {
        if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    }
    for (int len = 2; len <= lim; len <<= 1) {
        int m = len >> 1;
        int wn = qpow(3, (p - 1) / len, p);
        if (opt == -1) wn = qpow(wn, p - 2, p);
        for (int i = 0; i < lim; i += len) {
            int w = 1;
            for (int j = 0; j < m; ++j) {
                int u = a[i + j];
                int v = int(1LL * a[i + j + m] * w % p);
                a[i + j]       = (u + v < p ? u + v : u + v - p);
                a[i + j + m]   = (u - v >= 0 ? u - v : u - v + p);
                w = int(1LL * w * wn % p);
            }
        }
    }
    if (opt == -1) {
        int inv = qpow(lim, p - 2, p);
        for (int i = 0; i < lim; ++i) {
            a[i] = int(1LL * a[i] * inv % p);
        }
    }
}

struct NTTThreadArg {
    int *a, lim, opt, p;
};

void* ntt_thread_fn(void *vp) {
    auto *arg = (NTTThreadArg*)vp;
    ntt(arg->a, arg->lim, arg->opt, arg->p);
    return nullptr;
}

void poly_multiply(int *a,int *b,int *ab,int n,int p){
    memset(ab,0,sizeof(int)*(2*n-1));
    int lim=1; while(lim<2*n) lim<<=1;

    std::vector<int> A(lim,0), B(lim,0);
    for(int i=0;i<n;i++){ A[i]=a[i]; B[i]=b[i]; }

    if(lim <= 16){                  
        ntt(A.data(),lim,1,p);
        ntt(B.data(),lim,1,p);
    }else{
        pthread_t tA,tB;
        NTTThreadArg argA{A.data(),lim,1,p};
        NTTThreadArg argB{B.data(),lim,1,p};
        pthread_create(&tA,nullptr,ntt_thread_fn,&argA);
        pthread_create(&tB,nullptr,ntt_thread_fn,&argB);
        pthread_join(tA,nullptr);
        pthread_join(tB,nullptr);
    }

    for(int i=0;i<lim;i++)
        A[i]=int(1LL*A[i]*B[i]%p);

    ntt(A.data(),lim,-1,p);
    for(int i=0;i<2*n-1;i++) ab[i]=A[i];
}

int a[300000], b[300000], ab[300000];
int main(int argc, char *argv[]) {
    int test_begin = 0, test_end = 3;
    for (int i = test_begin; i <= test_end; ++i) {
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));

        auto Start = std::chrono::high_resolution_clock::now();
        poly_multiply(a, b, ab, n_, p_);
        auto End = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::ratio<1,1000>> elapsed = End - Start;
        fCheck(ab, n_, i);
        std::cout << "average latency for n = " << n_
                  << " p = " << p_
                  << " : " << elapsed.count() << " (us)\n";

        fWrite(ab, n_, i);
    }
    return 0;
}
