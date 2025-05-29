/*
 * ===========================================
 * 并行 NTT（collapse(2) OPenmp并行优化）
 * 编译：g++ -O2 -march=native -fopenmp main.cc -o ntt_omp_c2
 * ===========================================
 */
#include <bits/stdc++.h>
#include <omp.h>

/* ----------  I/O 与校验函数：原样保留  ---------- */
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

/* ----------  工具：快速幂 + 位反转  ---------- */
inline int qpow(int x,long long y,int p){
    long long res=1,base=x%p;
    while(y){ if(y&1)res=res*base%p; base=base*base%p; y>>=1; }
    return int(res);
}
inline void get_rev(int *rev,int lim){
    for(int i=0;i<lim;++i)
        rev[i]=(rev[i>>1]>>1)|((i&1)?(lim>>1):0);
}

/* ==================================================
 *            OpenMP NTT  (局部 REV + collapse(2))
 * ================================================== */
void ntt(int *a,int lim,int opt,int p){
    /* ---- 1. 位反转 ---- */
    std::vector<int> revv(lim);
    get_rev(revv.data(),lim);

    #pragma omp parallel for schedule(static)
    for(int i=0;i<lim;++i)
        if(i<revv[i]) std::swap(a[i],a[revv[i]]);

    /* ---- 2. 蝶形层 ---- */
    for(int len=2;len<=lim;len<<=1){
        int m=len>>1;
        int wn=qpow(3,(p-1)/len,p);
        if(opt==-1) wn=qpow(wn,p-2,p);

        /* 预生成 w^j 表，避免每段重复乘 */
        std::vector<int> wtab(m);
        wtab[0]=1;
        for(int j=1;j<m;++j) wtab[j]=1LL*wtab[j-1]*wn%p;

        int segCnt = lim/len;              // 段数
        #pragma omp parallel for collapse(2) schedule(static)
        for(int seg=0;seg<segCnt;++seg){
            for(int j=0;j<m;++j){
                int idx = seg*len + j;
                int u   = a[idx];
                int v   = 1LL * a[idx + m] * wtab[j] % p;
                a[idx]      = (u + v) % p;
                a[idx + m]  = (u - v + p) % p;
            }
        }
    }

    /* ---- 3. 逆变换除以 lim ---- */
    if(opt==-1){
        int inv=qpow(lim,p-2,p);
        #pragma omp parallel for schedule(static)
        for(int i=0;i<lim;++i)
            a[i]=1LL*a[i]*inv%p;
    }
}

/* ----------------- 多项式乘法 ----------------- */
void poly_multiply(int *a,int *b,int *ab,int n,int p){
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

/* ----------------- 主函数 ----------------- */
int a[300000],b[300000],ab[300000];
int main(){
    // 可显式控制线程：omp_set_num_threads(8);
    for(int id=0;id<=3;++id){
        int n,p;  fRead(a,b,&n,&p,id);
        auto st=std::chrono::high_resolution_clock::now();
        poly_multiply(a,b,ab,n,p);
        auto ed=std::chrono::high_resolution_clock::now();
        fCheck(ab,n,id);
        double ms=std::chrono::duration<double,std::milli>(ed-st).count();
        std::cout<<"n="<<n<<"  p="<<p<<"  time="<<ms<<" ms\n";
        fWrite(ab, n, id);
    }
    return 0;
}
