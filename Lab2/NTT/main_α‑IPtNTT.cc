// =============================================================
// main_ipt_ntt.cpp — α‑IPtNTT 多项式乘法（保持原始 I/O 接口）
// =============================================================
// 兼容原始 fRead / fCheck / fWrite。x86‑64 支持 AVX2 SIMD，加速三大典型模数；
// 其他架构自动退化为标量实现。已修复：
//   • ntt_scalar 逆变换循环类型错误 (u32 &x)
//   • pointwise_add → pointwise_acc，支持符号位
// =============================================================
#include <bits/stdc++.h>
#if defined(__x86_64__) || defined(_M_X64)
  #define ARCH_X86 1
  #include <immintrin.h>
#else
  #define ARCH_X86 0
#endif

using u32 = uint32_t; using u16 = uint16_t; using u64 = uint64_t;

// =============================================================
// 1. 原始 I/O 接口
// =============================================================
static void fRead(std::vector<u32>&a,std::vector<u32>&b,int &n,u32 &p,int id){
    std::ifstream fin("/nttdata/"+std::to_string(id)+".in");
    fin>>n>>p; a.resize(n); b.resize(n);
    for(int i=0;i<n;++i) fin>>a[i];
    for(int i=0;i<n;++i) fin>>b[i];
}
static void fCheck(const std::vector<u32>&c,int n,int id){
    std::ifstream fin("/nttdata/"+std::to_string(id)+".out"); u32 x;
    for(int i=0;i<2*n-1;++i){ fin>>x; if(x!=c[i]){ std::cout<<"多项式乘法结果错误\n"; return; } }
    std::cout<<"多项式乘法结果正确\n";
}
static void fWrite(const std::vector<u32>&c,int n,int id){
    std::ofstream fout("files/"+std::to_string(id)+".out");
    for(int i=0;i<2*n-1;++i) fout<<c[i]<<'\n';
}

// =============================================================
// 2. 工具函数
// =============================================================
static inline u32 mod_add(u32 a,u32 b,u32 p){ u32 s=a+b; return s>=p? s-p:s; }
static inline u32 mod_sub(u32 a,u32 b,u32 p){ return a>=b? a-b: a+p-b; }
static u32 qpow(u32 a,u64 e,u32 p){ u64 r=1,x=a; while(e){ if(e&1) r=r*x%p; x=x*x%p; e>>=1;} return (u32)r; }

static void build_rev(std::vector<u32>&rev,int lim){ int lg=__builtin_ctz(lim); rev.resize(lim); for(int i=0;i<lim;++i) rev[i]=(rev[i>>1]>>1)|((i&1)<<(lg-1)); }

// =============================================================
// 3. 标量 NTT
// =============================================================
static void ntt_scalar(std::vector<u32>&a,int lim,int opt,u32 p,u32 g){
    static std::vector<u32> rev; build_rev(rev,lim);
    for(int i=0;i<lim;++i) if(i<rev[i]) std::swap(a[i],a[rev[i]]);
    for(int len=2;len<=lim;len<<=1){
        int m=len>>1; u32 wn=qpow(g,(p-1)/len,p); if(opt==-1) wn=qpow(wn,p-2,p);
        for(int i=0;i<lim;i+=len){ u32 w=1;
            for(int j=0;j<m;++j){ u32 u=a[i+j], v=(u64)a[i+j+m]*w%p;
                a[i+j]=mod_add(u,v,p);
                a[i+j+m]=mod_sub(u,v,p);
                w=(u64)w*wn%p;
            }
        }
    }
    if(opt==-1){ u32 inv=qpow(lim,p-2,p); for(u32 &x:a) x=(u64)x*inv%p; }
}

// =============================================================
// 4. AVX2 16‑bit NTT（p < 2^15）
// =============================================================
#if ARCH_X86
static inline __m256i mont16(__m256i a,__m256i b,u16 p,u16 pinv){
    __m256i lo=_mm256_mullo_epi16(a,b);
    __m256i t=_mm256_mullo_epi16(lo,_mm256_set1_epi16(pinv));
    __m256i m=_mm256_mulhi_epu16(t,_mm256_set1_epi16(p));
    __m256i hi=_mm256_mulhi_epu16(a,b);
    return _mm256_sub_epi16(hi,m);
}
static inline __m256i barrett16(__m256i x,u16 p){
    // 近似 Barrett；两次比较足以 (p<32768)
    __m256i mask=_mm256_cmpgt_epi16(x,_mm256_set1_epi16(p-1));
    return _mm256_sub_epi16(x,_mm256_and_si256(mask,_mm256_set1_epi16(p)));
}
static void ntt_avx(std::vector<u16>&a,int lim,int opt,u16 p,u16 g){
    static std::vector<u32> rev; build_rev(rev,lim);
    for(int i=0;i<lim;++i) if(i<rev[i]) std::swap(a[i],a[rev[i]]);
    u16 pinv=((u32)1<<16)%p;
    for(int len=2;len<=lim;len<<=1){
        int m=len>>1; u16 wn=(u16)qpow(g,(p-1)/len,p); if(opt==-1) wn=(u16)qpow(wn,p-2,p);
        __m256i W=_mm256_set1_epi16(wn);
        for(int i=0;i<lim;i+=len){
            for(int j=0;j<m;j+=16){
                __m256i v1=_mm256_loadu_si256((__m256i*)&a[i+j]);
                __m256i v2=_mm256_loadu_si256((__m256i*)&a[i+j+m]);
                __m256i t=mont16(v2,W,p,pinv);
                __m256i sum=barrett16(_mm256_add_epi16(v1,t),p);
                __m256i dif=barrett16(_mm256_sub_epi16(v1,t),p);
                _mm256_storeu_si256((__m256i*)&a[i+j],sum);
                _mm256_storeu_si256((__m256i*)&a[i+j+m],dif);
            }
        }
    }
    if(opt==-1){ u16 inv=(u16)qpow(lim,p-2,p); __m256i INV=_mm256_set1_epi16(inv);
        for(int i=0;i<lim;i+=16){
            __m256i v=_mm256_loadu_si256((__m256i*)&a[i]);
            v=mont16(v,INV,p,pinv);
            v=barrett16(v,p);
            _mm256_storeu_si256((__m256i*)&a[i],v);
        }
    }
}
#endif

// =============================================================
// 5. α‑IPtNTT
// =============================================================
static int choose_alpha(int n,u32 p){ for(int a=0;a<=3;++a){ int mod=(a==0)? n: n>>(a-1); if(p%mod==1) return a; } return 0; }

template<typename Vec>
static inline void pointwise_acc(Vec &dst,const Vec&a,const Vec&b,u32 p,bool neg){
    int m=dst.size();
    if(!neg){
        for(int i=0;i<m;++i){ u32 t=((u64)a[i]*b[i])%p; dst[i]=dst[i]+t<p? dst[i]+t: dst[i]+t-p; }
    }else{
        for(int i=0;i<m;++i){ u32 t=((u64)a[i]*b[i])%p; dst[i]=dst[i]>=t? dst[i]-t: dst[i]+p-t; }
    }
}

template<typename T>
static void multiply_poly(const std::vector<u32>&A,const std::vector<u32>&B,std::vector<u32>&C,int n,u32 p){
    const u32 g=3; int alpha=choose_alpha(n,p), seg=n>>alpha, sp=1<<alpha;
    using Vec=std::vector<T>; std::vector<Vec> F(sp,Vec(seg)), G(sp,Vec(seg));
    for(int i=0;i<sp;++i) for(int j=0;j<seg;++j){ F[i][j]=A[i+j*sp]; G[i][j]=B[i+j*sp]; }

    // NTT
    #pragma omp parallel for schedule(static)
    for(int i=0;i<sp;++i){
        #if ARCH_X86
          if(sizeof(T)==2){ ntt_avx(*(std::vector<u16>*)&F[i],seg,1,(u16)p,(u16)g);
                             ntt_avx(*(std::vector<u16>*)&G[i],seg,1,(u16)p,(u16)g); }
          else{ ntt_scalar(*(std::vector<u32>*)&F[i],seg,1,p,g);
                ntt_scalar(*(std::vector<u32>*)&G[i],seg,1,p,g); }
        #else
          ntt_scalar(*(std::vector<u32>*)&F[i],seg,1,p,g);
          ntt_scalar(*(std::vector<u32>*)&G[i],seg,1,p,g);
        #endif
    }

    std::vector<Vec> H(sp,Vec(seg,0));
    for(int i=0;i<sp;++i){
        for(int l=0;l<=i;++l) pointwise_acc(H[i],F[l],G[i-l],p,false);
        for(int l=i+1;l<sp;++l) pointwise_acc(H[i],F[l],G[sp+i-l],p,true); // 乘 z=-1
    }

    // INTT
    #pragma omp parallel for schedule(static)
    for(int i=0;i<sp;++i){
        #if ARCH_X86
          if(sizeof(T)==2) ntt_avx(*(std::vector<u16>*)&H[i],seg,-1,(u16)p,(u16)g);
          else ntt_scalar(*(std::vector<u32>*)&H[i],seg,-1,p,g);
        #else
          ntt_scalar(*(std::vector<u32>*)&H[i],seg,-1,p,g);
        #endif
    }

    C.assign(2*n-1,0);
    for(int k=0;k<sp;++k) for(int j=0;j<seg;++j){ int idx=k+j*sp; C[idx]=mod_add(C[idx],H[k][j],p);} }

// =============================================================
// 6. 主程序
// =============================================================
int main(){
    int test_begin=0,test_end=3;
    for(int id=test_begin; id<=test_end; ++id){
        int n; u32 p; std::vector<u32>A,B; fRead(A,B,n,p,id);
        std::vector<u32>C;
        auto t0=std::chrono::high_resolution_clock::now();
        if(p<65536 && ARCH_X86){ multiply_poly<u16>(A,B,C,n,p); }
        else                    { multiply_poly<u32>(A,B,C,n,p); }
        auto t1=std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> dt=t1-t0;
        fCheck(C,n,id);
        std::cout<<"average latency for n="<<n<<" p="<<p<<" : "<<dt.count()*1000<<" (us)"<<std::endl;
        fWrite(C,n,id);
    }
    return 0;
}
