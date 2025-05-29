/*
 * ===========================================
 * OpenMP朴素优化NTT 
 * 编译：g++ -O2 -march=native -fopenmp main.cc -o ntt_omp_noreuse
 * ===========================================
 */
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <omp.h>

/* -------- 原 IO / check / write：保持不变 -------- */
void fRead(int *a,int *b,int *n,int *p,int id){
    std::string path="/nttdata/"+std::to_string(id)+".in";
    std::ifstream fin(path); fin>>*n>>*p;
    for(int i=0;i<*n;i++) fin>>a[i];
    for(int i=0;i<*n;i++) fin>>b[i];
}
void fCheck(int *ab,int n,int id){
    std::string path="/nttdata/"+std::to_string(id)+".out";
    std::ifstream fin(path);
    for(int i=0;i<2*n-1;i++){int x;fin>>x;if(x!=ab[i]){std::cout<<"结果错误\n";return;}}
    std::cout<<"结果正确\n";
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
/* ---------- 工具函数 ---------- */
inline int qpow(int x,long long y,int p){
    long long res=1,base=x%p;
    while(y){ if(y&1)res=res*base%p; base=base*base%p; y>>=1; }
    return int(res);
}
void get_rev(int *rev,int lim){
    for(int i=0;i<lim;++i)
        rev[i]=(rev[i>>1]>>1)|((i&1)?(lim>>1):0);
}

/* ---------- 并行 NTT (REV 每次新建) ---------- */
void ntt(int *a,int lim,int opt,int p){
    std::vector<int> revv(lim);
    get_rev(revv.data(),lim);

    #pragma omp parallel for schedule(static)
    for(int i=0;i<lim;++i)
        if(i<revv[i]) std::swap(a[i],a[revv[i]]);

    for(int len=2;len<=lim;len<<=1){
        int m=len>>1;
        int wn=qpow(3,(p-1)/len,p);
        if(opt==-1) wn=qpow(wn,p-2,p);

        #pragma omp parallel for schedule(static)
        for(int i=0;i<lim;i+=len){
            int w=1;
            for(int j=0;j<m;++j){
                int u=a[i+j];
                int v=1LL*a[i+j+m]*w%p;
                a[i+j]=(u+v)%p;
                a[i+j+m]=(u-v+p)%p;
                w=1LL*w*wn%p;
            }
        }
    }
    if(opt==-1){
        int inv=qpow(lim,p-2,p);
        #pragma omp parallel for schedule(static)
        for(int i=0;i<lim;++i) a[i]=1LL*a[i]*inv%p;
    }
}

/* ---------- 多项式乘法 ---------- */
void poly_multiply(int *a,int *b,int *ab,int n,int p){
    memset(ab, 0, sizeof(int) * (2 * n - 1));
    int lim=1; while(lim<2*n) lim<<=1;
    std::vector<int> A(lim,0),B(lim,0);

    for(int i=0;i<n;++i){ A[i]=a[i]; B[i]=b[i]; }

    ntt(A.data(),lim, 1,p);
    ntt(B.data(),lim, 1,p);

    for(int i=0;i<lim;++i)
        A[i]=1LL*A[i]*B[i]%p;

    ntt(A.data(),lim,-1,p);

    for(int i=0;i<2*n-1;++i) ab[i]=A[i];
}

/* ---------- 主函数 ---------- */
int a[300000],b[300000],ab[300000];
int main(){
    // omp_set_num_threads(8);  // 如需固定线程
    for(int id=0;id<=3;++id){
        int n,p; fRead(a,b,&n,&p,id);
        memset(ab, 0, sizeof(ab));
        auto st=std::chrono::high_resolution_clock::now();
        poly_multiply(a,b,ab,n,p);
        auto ed=std::chrono::high_resolution_clock::now();
        fCheck(ab,n,id);
        double ms=std::chrono::duration<double,std::milli>(ed-st).count();
        std::cout<<"n="<<n<<" p="<<p<<" time="<<ms<<" ms\n";
        fWrite(ab, n, id);
    }
    return 0;
}
