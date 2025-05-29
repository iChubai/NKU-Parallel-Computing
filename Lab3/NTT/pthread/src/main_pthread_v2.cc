/*
 * ===========================================
 * 文件名 main_ptread_v2.cc
 * 描述: NTT的静态线程+信号量同步
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
#define V2_THREADS 4

static int *glob_a, glob_len, glob_m, glob_wn, glob_p, glob_lim;
static sem_t sem_start[V2_THREADS], sem_done[V2_THREADS];

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

static void *v2_worker(void *arg){
    int tid=(long long)arg;
    for(;;){
        sem_wait(&sem_start[tid]);
        if(glob_len==0) break;
        for(int i=tid*glob_len;i<glob_lim;i+=V2_THREADS*glob_len){
            int w=1;
            for(int j=0;j<glob_m;++j){
                int u=glob_a[i+j];
                int v=1LL*glob_a[i+j+glob_m]*w%glob_p;
                glob_a[i+j]=(u+v)%glob_p;
                glob_a[i+j+glob_m]=(u-v+glob_p)%glob_p;
                w=1LL*w*glob_wn%glob_p;
            }
        }
        sem_post(&sem_done[tid]);
    }
    return NULL;
}

static void setup_v2_threads(){
    static bool inited=false;
    if(inited) return;
    for(int i=0;i<V2_THREADS;++i){ sem_init(&sem_start[i],0,0); sem_init(&sem_done[i],0,0); }
    for(long long i=0;i<V2_THREADS;++i) pthread_create(new pthread_t,0,v2_worker,(void*)i);
    inited=true;
}

static void ntt_v2(int *a,int lim,int opt,int p){
    int rev[lim]; get_rev(rev,lim);
    for(int i=0;i<lim;++i) if(i<rev[i]) std::swap(a[i],a[rev[i]]);
    setup_v2_threads();
    for(int len=2;len<=lim;len<<=1){
        glob_a=a; glob_len=len; glob_m=len>>1;
        glob_wn=qpow(3,(p-1)/len,p); if(opt==-1) glob_wn=qpow(glob_wn,p-2,p);
        glob_p=p; glob_lim=lim;
        for(int t=0;t<V2_THREADS;++t) sem_post(&sem_start[t]);
        for(int t=0;t<V2_THREADS;++t) sem_wait(&sem_done[t]);
    }
    if(opt==-1){
        int inv=qpow(lim,p-2,p);
        for(int i=0;i<lim;++i) a[i]=1LL*a[i]*inv%p;
    }
}

static void poly_multiply_v2(int *a,int *b,int *ab,int n,int p){
    memset(ab,0,sizeof(int)*(2*n-1));
    int lim=1; while(lim<2*n) lim<<=1;
    int *A=new int[lim](), *B=new int[lim]();
    memcpy(A,a,sizeof(int)*n); memcpy(B,b,sizeof(int)*n);
    ntt_v2(A,lim,1,p); ntt_v2(B,lim,1,p);
    for(int i=0;i<lim;++i) A[i]=1LL*A[i]*B[i]%p;
    ntt_v2(A,lim,-1,p);
    memcpy(ab,A,sizeof(int)*(2*n-1));
    delete[] A; delete[] B;
}

inline int mulmod(long long a,long long b,int p){return int(a*b%p);}

static int *gA,lenB,mB,wnB,wnStepB,pB,idxB,limB;

static void* slave(void*tid_){
    int tid=(long long)tid_;
    for(;;){
        sem_wait(&sem_start[tid]);
        if(lenB==0) break;
        int w=qpow(wnB,tid,pB);
        for(int j=tid;j<mB;j+=V2_THREADS){
            int u=gA[idxB+j],v=mulmod(gA[idxB+j+mB],w,pB);
            gA[idxB+j]=(u+v)%pB;
            gA[idxB+j+mB]=(u-v+pB)%pB;
            w=mulmod(w,wnStepB,pB);
        }
        sem_post(&sem_done[tid]);
    }
    return 0;
}

static void init_v2(){
    static bool ok=false; if(ok) return;
    for(int i=0;i<V2_THREADS;++i){sem_init(&sem_start[i],0,0);sem_init(&sem_done[i],0,0);}
    for(long long i=0;i<V2_THREADS;++i) pthread_create(new pthread_t,0,slave,(void*)i);
    ok=true;
}

static void ntt_v2_bfly(int *a,int lim,int opt,int p){
    std::unique_ptr<int[]> rev(new int[lim]); get_rev(rev.get(),lim);
    for(int i=0;i<lim;++i) if(i<rev[i]) std::swap(a[i],a[rev[i]]);
    init_v2();
    gA=a; limB=lim;
    for(int len=2;len<=lim;len<<=1){
        lenB=len; mB=len>>1; wnB=qpow(3,(p-1)/len,p);
        if(opt==-1) wnB=qpow(wnB,p-2,p);
        wnStepB=qpow(wnB,V2_THREADS,p); pB=p;
        for(int i=0;i<lim;i+=len){
            idxB=i;
            for(int t=0;t<V2_THREADS;++t) sem_post(&sem_start[t]);
            for(int t=0;t<V2_THREADS;++t) sem_wait(&sem_done[t]);
        }
    }
    if(opt==-1){int inv=qpow    (lim,p-2,p); for(int i=0;i<lim;++i)a[i]=mulmod(a[i],inv,p);}
}

void poly_multiply_v2bfly(int *a,int *b,int *ab,int n,int p){
    int lim=1; while(lim<2*n) lim<<=1;
    std::vector<int> A(lim,0),B(lim,0);
    std::memcpy(A.data(),a,sizeof(int)*n);
    std::memcpy(B.data(),b,sizeof(int)*n);
    ntt_v2_bfly(A.data(),lim,1,p); ntt_v2_bfly(B.data(),lim,1,p);
    for(int i=0;i<lim;++i) A[i]=mulmod(A[i],B[i],p);
    ntt_v2_bfly(A.data(),lim,-1,p);
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
        poly_multiply_v2(a, b, ab, n_, p_);
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
