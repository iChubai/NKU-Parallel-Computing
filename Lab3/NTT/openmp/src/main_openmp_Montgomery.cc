/*
 * ===========================================
 * 文件名: main_openmp_Montgomery.cc
 * 描述: 使用Montgomery规约实现OpenMP并行优化的NTT算法
 * 编译: g++ -O2 -march=native -fopenmp main_openmp_Montgomery.cc -o ntt_openmp_montgomery
 * ===========================================
 */
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <omp.h>

// Montgomery规约相关常量和函数
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

// Montgomery规约器
class MontgomeryReducer {
private:
    u32 mod;       
    u32 mod_inv;   
    u32 r2;        

public:
    MontgomeryReducer(u32 mod) : mod(mod) {
        mod_inv = inverse(mod, 1U << 31);  
        r2 = 0;
        u64 r = 1;
        for (int i = 0; i < 64; ++i) {
            r = (r << 1) % mod;
        }
        r2 = r;
    }

    // 单个数值的Montgomery规约
    u32 reduce(u64 x) {
        u32 q = (u32)x * mod_inv;  // q = x * mod' mod 2^32
        u64 m = (u64)q * mod;      // m = q * mod
        u32 y = (x - m) >> 32;     // y = (x - m) / 2^32
        return x < m ? y + mod : y; // 保证结果非负
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

/* -------- IO / check / write 函数：保持不变 -------- */
void fRead(int *a, int *b, int *n, int *p, int id){
    std::string path="/nttdata/"+std::to_string(id)+".in";
    std::ifstream fin(path); fin>>*n>>*p;
    for(int i=0;i<*n;i++) fin>>a[i];
    for(int i=0;i<*n;i++) fin>>b[i];
}

void fCheck(int *ab, int n, int id){
    std::string path="/nttdata/"+std::to_string(id)+".out";
    std::ifstream fin(path);
    for(int i=0;i<2*n-1;i++){int x;fin>>x;if(x!=ab[i]){std::cout<<"结果错误\n";return;}}
    std::cout<<"结果正确\n";
}

void fWrite(int *ab, int n, int input_id){
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

/* ---------- 工具函数 ---------- */
inline int qpow(int x, long long y, int p){
    long long res=1, base=x%p;
    while(y){ if(y&1)res=res*base%p; base=base*base%p; y>>=1; }
    return int(res);
}

void get_rev(int *rev, int lim){
    for(int i=0;i<lim;++i)
        rev[i]=(rev[i>>1]>>1)|((i&1)?(lim>>1):0);
}

/* ---------- 基于Montgomery规约的并行NTT ---------- */
void ntt_montgomery(int *a, int lim, int opt, int p){
    // 创建Montgomery规约器
    MontgomeryReducer reducer(p);
    
    // 位反转
    std::vector<int> revv(lim);
    get_rev(revv.data(), lim);

    #pragma omp parallel for schedule(static)
    for(int i=0;i<lim;++i)
        if(i<revv[i]) std::swap(a[i], a[revv[i]]);

    // 将所有数转换到Montgomery域
    #pragma omp parallel for schedule(static)
    for(int i=0;i<lim;++i)
        a[i] = reducer.to_montgomery(a[i]);

    // NTT蝴蝶操作
    for(int len=2;len<=lim;len<<=1){
        int m=len>>1;
        int wn=qpow(3, (p-1)/len, p);
        if(opt==-1) wn=qpow(wn, p-2, p);
        
        // 将单位根转换到Montgomery域
        wn = reducer.to_montgomery(wn);

        #pragma omp parallel for schedule(static)
        for(int i=0;i<lim;i+=len){
            u32 w = reducer.to_montgomery(1);  // Montgomery域中的1
            for(int j=0;j<m;++j){
                u32 u = a[i+j];
                u32 v = reducer.mul(a[i+j+m], w);
                
                // 模加和模减
                a[i+j] = u + v >= p ? u + v - p : u + v;
                a[i+j+m] = u >= v ? u - v : u + p - v;
                
                // 更新旋转因子
                w = reducer.mul(w, wn);
            }
        }
    }

    // 如果是逆变换，需要除以lim
    if(opt==-1){
        int inv = qpow(lim, p-2, p);
        inv = reducer.to_montgomery(inv);
        
        #pragma omp parallel for schedule(static)
        for(int i=0;i<lim;++i)
            a[i] = reducer.mul(a[i], inv);
    }

    // 将结果从Montgomery域转换回普通域
    #pragma omp parallel for schedule(static)
    for(int i=0;i<lim;++i)
        a[i] = reducer.from_montgomery(a[i]);
}

/* ---------- 基于Montgomery规约的多项式乘法 ---------- */
void poly_multiply_montgomery(int *a, int *b, int *ab, int n, int p){
    memset(ab, 0, sizeof(int) * (2 * n - 1));
    int lim=1; while(lim<2*n) lim<<=1;
    std::vector<int> A(lim, 0), B(lim, 0);

    for(int i=0;i<n;++i){ A[i]=a[i]; B[i]=b[i]; }

    ntt_montgomery(A.data(), lim, 1, p);
    ntt_montgomery(B.data(), lim, 1, p);

    // 创建Montgomery规约器用于点乘
    MontgomeryReducer reducer(p);
    
    // 将A和B转换到Montgomery域（虽然ntt_montgomery已经转换过，但这里为安全起见再次转换）
    #pragma omp parallel for schedule(static)
    for(int i=0;i<lim;++i){
        A[i] = reducer.to_montgomery(A[i]);
        B[i] = reducer.to_montgomery(B[i]);
    }
    
    // 在Montgomery域中进行点乘
    #pragma omp parallel for schedule(static)
    for(int i=0;i<lim;++i)
        A[i] = reducer.mul(A[i], B[i]);
    
    // 从Montgomery域转换回来
    #pragma omp parallel for schedule(static)
    for(int i=0;i<lim;++i)
        A[i] = reducer.from_montgomery(A[i]);

    ntt_montgomery(A.data(), lim, -1, p);

    for(int i=0;i<2*n-1;++i) ab[i]=A[i];
}

/* ---------- 优化版本：无需多余的Montgomery域转换 ---------- */
void poly_multiply_montgomery_optimized(int *a, int *b, int *ab, int n, int p){
    memset(ab, 0, sizeof(int) * (2 * n - 1));
    int lim=1; while(lim<2*n) lim<<=1;
    std::vector<int> A(lim, 0), B(lim, 0);

    for(int i=0;i<n;++i){ A[i]=a[i]; B[i]=b[i]; }

    ntt_montgomery(A.data(), lim, 1, p);
    ntt_montgomery(B.data(), lim, 1, p);

    // 创建Montgomery规约器用于点乘
    MontgomeryReducer reducer(p);
    
    // 点乘 - 注意这里我们不需要再次转换，因为ntt_montgomery已经返回了普通域的值
    #pragma omp parallel for schedule(static)
    for(int i=0;i<lim;++i)
        A[i] = (int)(((long long)A[i] * B[i]) % p);

    ntt_montgomery(A.data(), lim, -1, p);

    for(int i=0;i<2*n-1;++i) ab[i]=A[i];
}

/* ---------- 主函数 ---------- */
int a[300000], b[300000], ab[300000];
int main(){
    // omp_set_num_threads(8);  // 如需固定线程数
    for(int id=0;id<=3;++id){
        int n, p; fRead(a, b, &n, &p, id);
        memset(ab, 0, sizeof(ab));
        auto st=std::chrono::high_resolution_clock::now();
        
        // 使用优化版本
        poly_multiply_montgomery_optimized(a, b, ab, n, p);
        
        auto ed=std::chrono::high_resolution_clock::now();
        fCheck(ab, n, id);
        double ms=std::chrono::duration<double,std::milli>(ed-st).count();
        std::cout<<"n="<<n<<" p="<<p<<" time="<<ms<<" ms\n";
        fWrite(ab, n, id);
    }
    return 0;
} 