/*
 * ==========================================
 * radix-4 NTT Openmp优化
 *   • log₂N 偶数  → radix‑4 (digit‑reverse‑4 + 4‑叉蝶形)
 *   • log₂N 奇数  → 回退到可靠 radix‑2
 *   • OpenMP 并行
 * ==========================================
 */
#include <bits/stdc++.h>
#include <omp.h>

void fRead(int *a,int *b,int *n,int *p,int id){
    std::ifstream fin("/nttdata/"+std::to_string(id)+".in");
    fin>>*n>>*p;
    for(int i=0;i<*n;i++) fin>>a[i];
    for(int i=0;i<*n;i++) fin>>b[i];
}
void fCheck(int *ab,int n,int id){
    std::ifstream fin("/nttdata/"+std::to_string(id)+".out");
    for(int i=0;i<2*n-1;i++){int x;fin>>x; if(x!=ab[i]){std::cout<<"结果错误\n";return;}}
    std::cout<<"结果正确\n";
}

inline int qpow(long long x,long long y,int p){
    long long r=1%p; x%=p;
    for(;y;y>>=1,x=x*x%p) if(y&1) r=r*x%p;
    return int(r);
}

void bit_reverse(int *a,int n){
    int lg=__builtin_ctz(n);
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;i++){
        int rev=0;
        for(int j=0;j<lg;j++) if(i>>j&1) rev|=1<<(lg-1-j);
        if(i<rev) std::swap(a[i],a[rev]);
    }
}
    

void ntt_rad2(int *a,int n,bool inv,int p){
    static std::vector<int> rev;
    bit_reverse(a,n);                           // 置换
    for(int len=2; len<=n; len<<=1){
        int wn=qpow(3,(p-1)/len,p);
        if(inv) wn=qpow(wn,p-2,p);
        int half=len>>1;
        #pragma omp parallel for schedule(static)
        for(int blk=0; blk<n; blk+=len){
            int w=1;
            for(int j=0;j<half;j++){
                int u=a[blk+j];
                int v=1LL*a[blk+j+half]*w%p;
                a[blk+j]        =(u+v)%p;
                a[blk+j+half]   =(u-v+p)%p;
                w=1LL*w*wn%p;
            }
        }
    }
    if(inv){
        int invN=qpow(n,p-2,p);
        #pragma omp parallel for schedule(static)
        for(int i=0;i<n;i++) a[i]=1LL*a[i]*invN%p;
    }
}

void digrev4(int *a,int n){
    int pairs=__builtin_ctz(n)>>1;              // 2‑bit 块数
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;i++){
        int rev=0,tmp=i;
        for(int j=0;j<pairs;j++){ rev=(rev<<2)|(tmp&3); tmp>>=2;}
        if(i<rev) std::swap(a[i],a[rev]);
    }
}

void ntt_rad4(int *a,int n,bool inv,int p){
    digrev4(a,n);
    for(int len=4; len<=n; len<<=2){
        int m=len>>2;
        int wn=qpow(3,(p-1)/len,p);
        if(inv) wn=qpow(wn,p-2,p);
        int J=qpow(wn,m,p);                    // √‑1
        std::vector<int> wtab(m); wtab[0]=1;
        for(int j=1;j<m;j++) wtab[j]=1LL*wtab[j-1]*wn%p;

        #pragma omp parallel for schedule(static)
        for(int blk=0; blk<n; blk+=len){
            for(int j=0;j<m;j++){
                int w1=wtab[j];
                int w2=1LL*w1*w1%p;
                int w3=1LL*w2*w1%p;
                int A=a[blk+j];
                int B=1LL*a[blk+j+m]*w1%p;
                int C=1LL*a[blk+j+2*m]*w2%p;
                int D=1LL*a[blk+j+3*m]*w3%p;

                int T0=(A+C)%p, T1=(A+p-C)%p;
                int T2=(B+D)%p, T3=1LL*(B+p-D)*J%p;

                a[blk+j      ]=(T0+T2)%p;
                a[blk+j+m    ]=(T1+T3)%p;
                a[blk+j+2*m  ]=(T0+p-T2)%p;
                a[blk+j+3*m  ]=(T1+p-T3)%p;
            }
        }
    }
    if(inv){
        int invN=qpow(n,p-2,p);
        #pragma omp parallel for schedule(static)
        for(int i=0;i<n;i++) a[i]=1LL*a[i]*invN%p;
    }
}


void ntt(int *a,int n,bool inv,int p){
    bool odd = __builtin_ctz(n) & 1;
    if(odd)  ntt_rad2(a,n,inv,p);          // 奇数阶 → radix‑2 全程
    else     ntt_rad4(a,n,inv,p);          // 偶数阶 → radix‑4
}

void poly_mul(int *a,int *b,int *c,int n,int p){
    int N=1; while(N<2*n) N<<=1;
    std::vector<int> A(N),B(N);
    for(int i=0;i<n;i++){A[i]=a[i]; B[i]=b[i];}
    ntt(A.data(),N,false,p);
    ntt(B.data(),N,false,p);
    for(int i=0;i<N;i++) A[i]=1LL*A[i]*B[i]%p;
    ntt(A.data(),N,true,p);
    for(int i=0;i<2*n-1;i++) c[i]=A[i];
}

int a[300000],b[300000],ab[300000];
int main(){
    for(int id=0;id<=3;++id){
        int n,p; fRead(a,b,&n,&p,id);
        auto st=std::chrono::high_resolution_clock::now();
        poly_mul(a,b,ab,n,p);
        auto ed=std::chrono::high_resolution_clock::now();
        fCheck(ab,n,id);
        std::cout<<"n="<<n<<"  p="<<p<<"  time="
                 <<std::chrono::duration<double,std::milli>(ed-st).count()
                 <<" ms\n";
    }
    return 0;
}
