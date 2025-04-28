/*
 * ===========================================
 * 文件名: main_DIF_DIT.cc
 * 描述: 使用DIF DIT实现NTT的串行算法
 * ===========================================
 */


#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <algorithm>

typedef long long ll;

// 保留原有的文件读写函数
void fRead(int *a, int *b, int *n, int *p, int input_id) {
    std::string str1 = "/nttdata/";
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

void fCheck(int *ab, int n, int input_id) {
    std::string str1 = "/nttdata/";
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
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

void fWrite(int *ab, int n, int input_id) {
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

// NTT相关实现
inline ll fpow(ll a, ll b, int P) {
    ll res = 1; a %= P;
    for (; b; b >>= 1) {
        if (b & 1) (res *= a) %= P;
        (a *= a) %= P;
    }
    return res;
}

void calc_powg(int w[], int G, int P, int gen) {
    w[0] = 1; ll f;
    const int g = fpow(gen, (P-1)/G, P);
    for (int t = 0; (1<<(t+1)) < G; ++t) {
        f = w[1<<t] = fpow(g, G>>(t+2), P);
        for (int x = 1<<t; x < 1<<(t+1); ++x)
            w[x] = (ll)f * w[x - (1<<t)] % P;
    }
}

void DIF(int f[], int l, int P, int w[]) {
    int lim = 1 << l;
    ll g, h;
    for (int len = lim; len > 1; len >>= 1) {
        for (int st = 0, t = 0; st < lim; st += len, ++t) {
            for (int i = st; i < st + len/2; ++i) {
                g = f[i];
                h = (ll)f[i + len/2] * w[t] % P;
                f[i] = (g + h) % P;
                f[i + len/2] = (P + g - h) % P;
            }
        }
    }
}

void DIT(int f[], int l, int P, int w[]) {
    int lim = 1 << l;
    ll g, h;
    for (int len = 2; len <= lim; len <<= 1) {
        for (int st = 0, t = 0; st < lim; st += len, ++t) {
            for (int i = st; i < st + len/2; ++i) {
                g = f[i];
                h = f[i + len/2];
                f[i] = (g + h) % P;
                f[i + len/2] = (P + g - h) * w[t] % P;
            }
        }
    }
    const ll invl = fpow(lim, P-2, P);
    for (int i = 0; i < lim; ++i)
        f[i] = invl * f[i] % P;
    std::reverse(f + 1, f + lim);
}

void poly_multiply_optimized(int *a, int *b, int *ab, int n, int p, int gen = 3) {
    memset(ab, 0, sizeof(int) * (2 * n - 1));
    int l = 0;
    while ((1 << l) < 2 * n) l++;
    int lim = 1 << l;
    
    int *A = new int[lim]();
    int *B = new int[lim]();
    int *w = new int[lim]();
    
    for (int i = 0; i < n; ++i) {
        A[i] = a[i];
        B[i] = b[i];
    }
    
    calc_powg(w, lim, p, gen);
    DIF(A, l, p, w);
    DIF(B, l, p, w);
    
    for (int i = 0; i < lim; ++i)
        A[i] = (ll)A[i] * B[i] % p;
    
    DIT(A, l, p, w);
    
    for (int i = 0; i < 2 * n - 1; ++i)
        ab[i] = A[i];
    
    delete[] A;
    delete[] B;
    delete[] w;
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
        
        // 使用优化的多项式乘法
        poly_multiply_optimized(a, b, ab, n_, p_);
        
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<std::endl;
        fWrite(ab, n_, i);
    }
    return 0;
}
