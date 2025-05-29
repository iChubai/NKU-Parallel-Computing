/*
 * ===========================================
 * 文件名 main.cc
 * 描述: NTT的串行实现
 * ===========================================
 */
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
// 可以自行添加需要的头文件

void fRead(int *a, int *b, int *n, int *p, int input_id){
    // 数据输入函数
    std::string str1 = "../nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    fin>>*n>>*p;
    for (int i = 0; i < *n; i++){
        fin>>a[i];
    }
    for (int i = 0; i < *n; i++){
        fin>>b[i];
    }
}

void fCheck(int *ab, int n, int input_id){
    // 判断多项式乘法结果是否正确
    std::string str1 = "../nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++){
        int x;
        fin>>x;
        // std::cout<<"i= "<<i<<' '<<"x: "<<x<<' '<<"ab: "<<ab[i]<<std::endl;
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

void fWrite(int *ab, int n, int input_id){
    // 数据输出函数, 可以用来输出最终结果, 也可用于调试时输出中间数组
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++){
        fout<<ab[i]<<'\n';
    }
}

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

int a[300000], b[300000], ab[300000];
int main(int argc, char *argv[])
{

    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入模数分别为 7340033 104857601 469762049 263882790666241
    // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
    // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1
    int test_begin = 0;
    int test_end = 3;
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();
        // 使用NTT实现的多项式乘法
        poly_multiply(a, b, ab, n_, p_);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<std::endl;
        // 可以使用 fWrite 函数将 ab 的输出结果打印到 files 文件夹下
        // 禁止使用 cout 一次性输出大量文件内容
        fWrite(ab, n_, i);
    }
    return 0;
}
