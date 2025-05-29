/* =========================================================
 * three‑mod NTT CRT – OpenMP 并行 
 * compile:  g++ -O2 -march=native -fopenmp main_CRT_openmp.cc -o ntt
 * =========================================================*/
#include <bits/stdc++.h>
#include <omp.h>
void fRead(int *a, int *b, int *n, long long *p, int input_id){   
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin(data_path);
    fin >> *n >> *p;                                              
    for(int i=0;i<*n;i++) fin >> a[i];
    for(int i=0;i<*n;i++) fin >> b[i];
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

/* ---------- 快速幂 ---------- */
inline int qpow(long long x,long long y,int p){
    long long r=1%p; x%=p;
    for(;y;y>>=1,x=x*x%p) if(y&1) r=r*x%p;
    return int(r);
}

/* ---------- LUT 模数 ---------- */
static constexpr int PRIMES[3]={469762049,104857601,7340033};

/* ---------- 预处理 plan ---------- */
struct Plan{
    std::vector<int> rev;
    std::vector<int> wn;     // 根表按层存
} plan[3];

void build_plan(int mod,int idx,int maxLim){
    int lv=__builtin_ctz(maxLim);
    plan[idx].rev.resize(maxLim);
    for(int i=0;i<maxLim;i++)
        plan[idx].rev[i]=(plan[idx].rev[i>>1]>>1)|((i&1)?(maxLim>>1):0);
    plan[idx].wn.resize(lv);
    for(int k=0,len=2;k<lv;k++,len<<=1)
        plan[idx].wn[k]=qpow(3,(mod-1)/len,mod);
}

/* ---------- 单模 radix‑2 NTT (OpenMP) ---------- */
void ntt_omp(int *a,int lim,bool inv,int mod,int idx){
    const auto &rev=plan[idx].rev;
    #pragma omp parallel for schedule(static)
    for(int i=0;i<lim;i++) if(i<rev[i]) std::swap(a[i],a[rev[i]]);

    constexpr int BLOCK=256*1024;               // 1 MiB
    for(int base=0;base<lim;base+=BLOCK){
        int seg=std::min(BLOCK,lim-base);
        int lv=0;
        for(int len=2;len<=seg;len<<=1,++lv){
            int half=len>>1;
            int wn=plan[idx].wn[lv]; if(inv) wn=qpow(wn,mod-2,mod);

            if(half>=512){
                #pragma omp parallel
                {
                    #pragma omp single nowait
                    for(int blk=base; blk<base+seg; blk+=len){
                        #pragma omp task firstprivate(blk,wn,half,mod,a)
                        {
                            int w=1;
                            for(int j=0;j<half;j++){
                                int u=a[blk+j];
                                int v=int(1LL*a[blk+j+half]*w%mod);
                                a[blk+j]        = u+v<mod?u+v:u+v-mod;
                                a[blk+j+half]   = u-v>=0?u-v:u-v+mod;
                                w=int(1LL*w*wn%mod);
                            }
                        }
                    }
                }           // implicit barrier
            }else{
                #pragma omp parallel for schedule(static) collapse(2)
                for(int blk=base; blk<base+seg; blk+=len)
                    for(int j=0;j<half;j++){
                        int w=qpow(wn,j,mod);
                        int u=a[blk+j];
                        int v=int(1LL*a[blk+j+half]*w%mod);
                        a[blk+j]        = u+v<mod?u+v:u+v-mod;
                        a[blk+j+half]   = u-v>=0?u-v:u-v+mod;
                    }
            }
        }
    }
    if(inv){
        int invLim=qpow(lim,mod-2,mod);
        #pragma omp parallel for schedule(static)
        for(int i=0;i<lim;i++) a[i]=int(1LL*a[i]*invLim%mod);
    }
}

/* ---------- 选择模数 ---------- */
std::vector<int> choose_mod(long long p){
    for(int i=0;i<3;i++) if(p==PRIMES[i]) return {i};
    return {0,1,2};
}

/* ---------- 多项式卷积 + CRT ---------- */
void poly_conv(const int *A,const int *B,int n,long long P,int *out){
    auto idx=choose_mod(P);
    int lim=1; while(lim<2*n) lim<<=1;
    static std::unique_ptr<int[]> buf(new int[3*600000]);
    int *w0=buf.get(), *w1=w0+lim, *w2=w1+lim;
    int *work[3]={w0,w1,w2};

    /* ---- NTT for each mod ---- */
    #pragma omp parallel for
    for(int k=0;k<(int)idx.size();k++){
        int id=idx[k], mod=PRIMES[id];
        int *fa=work[id], *fb=fa+lim;
        std::fill(fa,fa+2*lim,0);
        for(int i=0;i<n;i++){ fa[i]=A[i]%mod; fb[i]=B[i]%mod; }
        ntt_omp(fa,lim,false,mod,id);
        ntt_omp(fb,lim,false,mod,id);
        for(int i=0;i<lim;i++) fa[i]=int(1LL*fa[i]*fb[i]%mod);
        ntt_omp(fa,lim,true ,mod,id);
    }

    if(idx.size()==1){
        int id=idx[0], mod=PRIMES[id];
        for(int i=0;i<2*n-1;i++) out[i]=work[id][i]%P;
        return;
    }

    /* ---- CRT merge ---- */
    const int *r0=work[idx[0]],
              *r1=work[idx[1]],
              *r2=(idx.size()==3)? work[idx[2]]:nullptr;
    long long p0=PRIMES[idx[0]], p1=PRIMES[idx[1]],
              p2=(idx.size()==3)?PRIMES[idx[2]]:1;
    long long T=P,M0=(p1%T)*(p2%T)%T,
                     M1=(p0%T)*(p2%T)%T,
                     M2=(p0%T)*(p1%T)%T;
    int y0=qpow(M0%p0,p0-2,p0),
        y1=qpow(M1%p1,p1-2,p1),
        y2=(idx.size()==3)?qpow(M2%p2,p2-2,p2):0;

    #pragma omp parallel for schedule(static)
    for(int i=0;i<2*n-1;i++){
        long long x=(1LL*r0[i]*M0%T*y0)%T;
        x=(x+1LL*r1[i]*M1%T*y1)%T;
        if(idx.size()==3) x=(x+1LL*r2[i]*M2%T*y2)%T;
        out[i]=int(x);
    }
}

/* ---------- I/O 缓冲 ---------- */
static int A[300000],B[300000],RES[600000];

int main(){
    build_plan(PRIMES[0],0,262144);
    build_plan(PRIMES[1],1,262144);
    build_plan(PRIMES[2],2,262144);

    for(int id=0; id<4; ++id){
        int n; long long P; fRead(A,B,&n,&P,id);
        auto st=std::chrono::high_resolution_clock::now();
        poly_conv(A,B,n,P,RES);
        auto ed=std::chrono::high_resolution_clock::now();
        fCheck(RES,n,id);
        std::cout<<"n="<<n<<" p="<<P<<" "
                 <<"time="<<std::chrono::duration<double,std::milli>(ed-st).count()
                 <<" ms\n";
        fWrite(RES,n,id);
    }
    return 0;
}
