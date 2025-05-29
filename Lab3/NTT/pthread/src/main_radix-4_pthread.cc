/*
 * ==========================================================
 * pthread 并行  Radix‑4 / Radix‑2  NTT  (安全版)
 *   · log₂N 偶数 → digit‑reverse‑4 + radix‑4
 *   · log₂N 奇数 → bit‑reverse     + radix‑2
 *   · 每个并行 for:  create‑join T 条 pthread
 * 编译: g++ -O2 -march=native -pthread main.cc -o ntt_pth
 * ==========================================================
 */
#include <bits/stdc++.h>
#include <pthread.h>

inline int qpow(long long x,long long y,int p){
    long long r=1%p; x%=p;
    for(; y; y>>=1, x=x*x%p) if(y&1) r=r*x%p;
    return int(r);
}

struct LoopArg{
    long long beg,end,step;
    const std::function<void(long long)> *body;   // 安全的堆指针
};
void* loop_worker(void* arg){
    auto* A = static_cast<LoopArg*>(arg);
    for(long long i=A->beg;i<A->end;i+=A->step) (*(A->body))(i);
    return nullptr;
}

template<typename F>
inline void parallel_for(long long beg,long long end,long long step,
                         const F& func,int THREADS)
{
    if(end - beg <= step * THREADS){          // 小任务串行
        for(long long i=beg;i<end;i+=step) func(i);
        return;
    }
    auto* body = new std::function<void(long long)>(func);   // 堆上
    long long chunk = ((end - beg) + THREADS - 1)/THREADS;
    std::vector<pthread_t> threads(THREADS);
    std::vector<LoopArg>   args(THREADS);

    for(int t=0;t<THREADS;++t){
        long long l = beg + t*chunk;
        long long r = std::min<long long>(end, l + chunk);
        args[t] = { l, r, step, body };
        pthread_create(&threads[t], nullptr, loop_worker, &args[t]);
    }
    for(auto& th : threads) pthread_join(th, nullptr);
    delete body;                                           // 回收
}

/* ---------- digit‑reverse‑4 ---------- */
void digrev4(int *a,int n,int T){
    int pairs = __builtin_ctz(n)>>1;
    parallel_for(0,n,1,[=](long long idx){
        int i=int(idx), rev=0, tmp=i;
        for(int j=0;j<pairs;j++){ rev=(rev<<2)|(tmp&3); tmp>>=2; }
        if(i<rev) std::swap(a[i],a[rev]);
    },T);
}

/* ---------- bit‑reverse (radix‑2) ---------- */
void bitrev(int *a,int n,int T){
    int lg=__builtin_ctz(n);
    parallel_for(0,n,1,[=](long long idx){
        int i=int(idx), rev=0;
        for(int j=0;j<lg;j++) if(i>>j&1) rev|=1<<(lg-1-j);
        if(i<rev) std::swap(a[i],a[rev]);
    },T);
}

/* ---------- radix‑2 NTT ---------- */
void ntt_rad2(int *a,int n,bool inv,int p,int T){
    bitrev(a,n,T);
    for(int len=2; len<=n; len<<=1){
        int wn=qpow(3,(p-1)/len,p);
        if(inv) wn=qpow(wn,p-2,p);
        int h=len>>1;
        parallel_for(0,n,len,[=](long long blk){
            int w=1;
            for(int j=0;j<h;++j){
                int u=a[blk+j];
                int v=1LL*a[blk+j+h]*w%p;
                a[blk+j]=(u+v)%p;
                a[blk+j+h]=(u-v+p)%p;
                w=1LL*w*wn%p;
            }
        },T);
    }
    if(inv){
        int invN=qpow(n,p-2,p);
        parallel_for(0,n,1,[=](long long i){ a[i]=1LL*a[i]*invN%p; },T);
    }
}

/* ---------- radix‑4 NTT ---------- */
void ntt_rad4(int *a,int n,bool inv,int p,int T){
    digrev4(a,n,T);
    for(int len=4; len<=n; len<<=2){
        int m=len>>2;
        int wn=qpow(3,(p-1)/len,p);
        if(inv) wn=qpow(wn,p-2,p);
        int J=qpow(wn,m,p);

        std::vector<int> wtab(m); wtab[0]=1;
        for(int j=1;j<m;++j) wtab[j]=1LL*wtab[j-1]*wn%p;

        parallel_for(0,n,len,[&](long long blk){
            for(int j=0;j<m;++j){
                int w1=wtab[j],
                    w2=1LL*w1*w1%p,
                    w3=1LL*w2*w1%p;
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
        },T);
    }
    if(inv){
        int invN=qpow(n,p-2,p);
        parallel_for(0,n,1,[=](long long i){ a[i]=1LL*a[i]*invN%p; },T);
    }
}

/* ---------- NTT 统一接口 ---------- */
void ntt(int *a,int n,bool inv,int p,int T){
    bool odd = __builtin_ctz(n)&1;
    if(odd) ntt_rad2(a,n,inv,p,T);
    else    ntt_rad4(a,n,inv,p,T);
}

/* ---------- 多项式乘法 ---------- */
void poly_mul(int *A,int *B,int *C,int n,int p,int T){
    int N=1; while(N<2*n) N<<=1;
    std::vector<int>a(N),b(N);
    parallel_for(0,n,1,[&](long long i){ a[i]=A[i]; b[i]=B[i]; },T);

    ntt(a.data(),N,false,p,T);
    ntt(b.data(),N,false,p,T);
    parallel_for(0,N,1,[&](long long i){ a[i]=1LL*a[i]*b[i]%p; },T);
    ntt(a.data(),N,true,p,T);

    parallel_for(0,2*n-1,1,[&](long long i){ C[i]=a[i]; },T);
}

/* ---------- IO ---------- */
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

/* ---------- 主函数 ---------- */
int a[300000],b[300000],ab[300000];
int main(){
    int THREADS = std::thread::hardware_concurrency();
    if(THREADS==0) THREADS=8;

    for(int id=0; id<=3; ++id){
        int n,p; fRead(a,b,&n,&p,id);
        auto st = std::chrono::high_resolution_clock::now();
        poly_mul(a,b,ab,n,p,THREADS);
        auto ed = std::chrono::high_resolution_clock::now();
        fCheck(ab,n,id);
        std::cout<<"n="<<n<<" p="<<p<<" threads="<<THREADS
                 <<" time="<<std::chrono::duration<double,std::milli>(ed-st).count()
                 <<" ms\n";
    }
    return 0;
}
