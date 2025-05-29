/*
 * ===========================================
 * 文件名 main_pthread_v1.cc
 * 描述: NTT的动态回收创建线程
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
#include <semaphore.h>
#include <vector>
#define V1_THREADS 4

void fRead(int *a, int *b, int *n, int *p, int input_id){
    // 数据输入函数
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

void fCheck(int *ab, int n, int input_id){
    // 判断多项式乘法结果是否正确
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
    memset(rev, 0, sizeof(int) * lim);
    for (int i = 0; i < lim; ++i) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) ? (lim >> 1) : 0);
    }
}


typedef struct {
    int *a; int len; int m; int wn; int p; int tid; int step; int lim;
} v1_arg_t;

static void *v1_kernel(void *arg){
    v1_arg_t *pa = (v1_arg_t*)arg;
    int *a=pa->a, len=pa->len, m=pa->m, wn=pa->wn, p=pa->p, tid=pa->tid, step=pa->step, lim=pa->lim;
    for(int i = tid*len; i < lim; i += step*len){
        int w = 1;
        for(int j=0;j<m;++j){
            int u=a[i+j], v=1LL*a[i+j+m]*w%p;
            a[i+j]=(u+v)%p;
            a[i+j+m]=(u-v+p)%p;
            w=1LL*w*wn%p;
        }
    }
    return NULL;
}

static void ntt_v1(int *a,int lim,int opt,int p){
    int rev[lim]; get_rev(rev,lim);
    for(int i=0;i<lim;++i) if(i<rev[i]) std::swap(a[i],a[rev[i]]);
    for(int len=2;len<=lim;len<<=1){
        int m=len>>1;
        int wn=qpow(3,(p-1)/len,p);
        if(opt==-1) wn=qpow(wn,p-2,p);
        pthread_t th[V1_THREADS];
        v1_arg_t param[V1_THREADS];
        for(int t=0;t<V1_THREADS;++t){
            param[t]={a,len,m,wn,p,t,V1_THREADS,lim};
            pthread_create(&th[t],0,v1_kernel,&param[t]);
        }
        for(int t=0;t<V1_THREADS;++t) pthread_join(th[t],0);
    }
    if(opt==-1){
        int inv=qpow(lim,p-2,p);
        for(int i=0;i<lim;++i) a[i]=1LL*a[i]*inv%p;
    }
}

static void poly_multiply_v1(int *a,int *b,int *ab,int n,int p){
    memset(ab,0,sizeof(int)*(2*n-1));
    int lim=1; while(lim<2*n) lim<<=1;
    int *A=new int[lim](), *B=new int[lim]();
    memcpy(A,a,sizeof(int)*n);
    memcpy(B,b,sizeof(int)*n);
    ntt_v1(A,lim,1,p); ntt_v1(B,lim,1,p);
    for(int i=0;i<lim;++i) A[i]=1LL*A[i]*B[i]%p;
    ntt_v1(A,lim,-1,p);
    memcpy(ab,A,sizeof(int)*(2*n-1));
    delete[] A; delete[] B;
}


inline int mulmod(long long a,long long b,int p){return int(a*b%p);}

struct Task {
    int *a, len, m, wn, p, tid, lim;
};

static void *bfly_worker(void *arg) {
    Task *t = (Task *)arg;
    int blockStep = t->len * V1_THREADS;        // 每个线程跨几个 block
    for (int base = t->tid * t->len; base < t->lim; base += blockStep) {
        int w = 1;
        for (int j = 0; j < t->m; ++j) {
            int u = t->a[base + j];
            int v = 1LL * t->a[base + j + t->m] * w % t->p;
            t->a[base + j]          = (u + v) % t->p;
            t->a[base + j + t->m]   = (u - v + t->p) % t->p;
            w = 1LL * w * t->wn % t->p;
        }
    }
    return nullptr;
}

static void ntt_v1_bfly(int *a, int lim, int opt, int p) {
    std::unique_ptr<int[]> rev(new int[lim]);
    get_rev(rev.get(), lim);
    for (int i = 0; i < lim; ++i)
        if (i < rev[i]) std::swap(a[i], a[rev[i]]);

    pthread_t th[V1_THREADS];
    Task task[V1_THREADS];

    for (int len = 2; len <= lim; len <<= 1) {
        int m  = len >> 1;
        int wn = qpow(3, (p - 1) / len, p);
        if (opt == -1) wn = qpow(wn, p - 2, p);

        for (int t = 0; t < V1_THREADS; ++t) {
            task[t] = {a, len, m, wn, p, t, lim};
            pthread_create(&th[t], nullptr, bfly_worker, &task[t]);
        }
        for (int t = 0; t < V1_THREADS; ++t)
            pthread_join(th[t], nullptr);
    }

    if (opt == -1) {
        int inv = qpow(lim, p - 2, p);
        for (int i = 0; i < lim; ++i) a[i] = 1LL * a[i] * inv % p;
    }
}


void poly_multiply_v1bfly(int *a,int *b,int *ab,int n,int p){
    int lim=1; while(lim<2*n) lim<<=1;
    std::vector<int> A(lim,0),B(lim,0);
    std::memcpy(A.data(),a,sizeof(int)*n);
    std::memcpy(B.data(),b,sizeof(int)*n);
    ntt_v1_bfly(A.data(),lim,1,p); ntt_v1_bfly(B.data(),lim,1,p);
    for(int i=0;i<lim;++i) A[i]=mulmod(A[i],B[i],p);
    ntt_v1_bfly(A.data(),lim,-1,p);
    std::memcpy(ab,A.data(),sizeof(int)*(2*n-1));
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
        poly_multiply_v1(a, b, ab, n_, p_);
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
