// =============================================================
// main_DIF_DIT_neon_with_Barrett.cc      (k = 40  Barrett, 存在问题)
// =============================================================
#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

using ll = long long;
#define VECW 4

/* ------------ k=40 Barrett 常量 & 标量 ------------ */
static inline uint64_t mu40(int P) {          // μ = floor(2^40 / P)
    return (1ULL << 40) / (uint32_t)P;
}
static inline uint32_t mul_mod_b40(uint32_t a,uint32_t b,int P,uint64_t MU){
    uint64_t x = uint64_t(a) * b;             // < 2^63
    uint64_t q = (x * MU) >> 40;              // ≈ x / P
    uint32_t r = uint32_t(x - q * P);         // 0‥2P
    if (r >= (uint32_t)P) r -= P;
    return r;
}

/* ------------ 向量版：4 lane 顺序 Barrett ------------ */
static inline int32x4_t mul_vec_b40(int32x4_t A,int32x4_t B,int P,uint64_t MU){
    int32x4_t R;
    for(int lane=0;lane<VECW;++lane){
        uint32_t a=vgetq_lane_u32(vreinterpretq_u32_s32(A),lane);
        uint32_t b=vgetq_lane_u32(vreinterpretq_u32_s32(B),lane);
        uint32_t r=mul_mod_b40(a,b,P,MU);
        R=vsetq_lane_s32(int32_t(r),R,lane);
    }
    return R;
}

/* ------------ 加减模（与前相同） ------------ */
inline int32x4_t mod_add_vec(int32x4_t a,int32x4_t b,int32x4_t P){
    int32x4_t s=vaddq_s32(a,b);
    int32x4_t ge=vreinterpretq_s32_u32(vcgeq_s32(s,P));
    return vsubq_s32(s,vandq_s32(ge,P));
}
inline int32x4_t mod_sub_vec(int32x4_t a,int32x4_t b,int32x4_t P){
    int32x4_t d=vsubq_s32(a,b);
    int32x4_t lt=vreinterpretq_s32_u32(vcltq_s32(d,vdupq_n_s32(0)));
    return vaddq_s32(d,vandq_s32(lt,P));
}

/* ------------ 幂表 ------------ */
static inline ll fpow(ll a,ll b,int P){ ll r=1;for(a%=P;b;b>>=1,a=a*a%P) if(b&1) r=r*a%P; return r;}
void calc_powg(int* w,int G,int P,int g){
    w[0]=1; int root=fpow(g,(P-1)/G,P);
    for(int t=0;(1<<(t+1))<G;++t){
        ll f=w[1<<t]=fpow(root,G>>(t+2),P);
        int blk=1<<t;
        for(int x=blk;x<(blk<<1);++x) w[x]=ll(f)*w[x-blk]%P;
    }
}

/* ------------ DIF NTT ------------ */
void DIF(int* f,int l,int P,const int* w){
    int32x4_t Pvec=vdupq_n_s32(P);
    uint64_t MU=mu40(P);
    int lim=1<<l;
    for(int len=lim;len>1;len>>=1){
        int half=len>>1;
        for(int st=0,t=0;st<lim;st+=len,++t){
            int32x4_t wt=vdupq_n_s32(w[t]);
            int i=st;
            for(;i+VECW<=st+half;i+=VECW){
                int32x4_t g=vld1q_s32(f+i);
                int32x4_t h=vld1q_s32(f+i+half);
                h=mul_vec_b40(h,wt,P,MU);
                vst1q_s32(f+i,      mod_add_vec(g,h,Pvec));
                vst1q_s32(f+i+half, mod_sub_vec(g,h,Pvec));
            }
            for(;i<st+half;++i){
                uint32_t h=mul_mod_b40(f[i+half],w[t],P,MU);
                uint32_t g=f[i];
                f[i]      =(g+h)%P;
                f[i+half] =(g+P-h)%P;
            }
        }
    }
}

/* ------------ DIT NTT + 归一化 ------------ */
void DIT(int* f,int l,int P,const int* w){
    int32x4_t Pvec=vdupq_n_s32(P);
    uint64_t MU=mu40(P);
    int lim=1<<l;
    for(int len=2;len<=lim;len<<=1){
        int half=len>>1;
        for(int st=0,t=0;st<lim;st+=len,++t){
            int32x4_t wt=vdupq_n_s32(w[t]);
            int i=st;
            for(;i+VECW<=st+half;i+=VECW){
                int32x4_t g=vld1q_s32(f+i);
                int32x4_t h=vld1q_s32(f+i+half);
                int32x4_t sum=mod_add_vec(g,h,Pvec);
                int32x4_t dif=mul_vec_b40(mod_sub_vec(g,h,Pvec),wt,P,MU);
                vst1q_s32(f+i,      sum);
                vst1q_s32(f+i+half, dif);
            }
            for(;i<st+half;++i){
                uint32_t g=f[i], h=f[i+half];
                uint32_t sum=g+h; if(sum>=uint32_t(P)) sum-=P;
                uint32_t dif=mul_mod_b40((g+P-h)%P,w[t],P,MU);
                f[i]=sum; f[i+half]=dif;
            }
        }
    }
    /* 乘 inv(lim) */
    uint32_t inv=fpow(lim,P-2,P);
    int32x4_t inv_vec=vdupq_n_s32(inv);
    int i=0;
    for(;i+VECW<=lim;i+=VECW){
        int32x4_t v=vld1q_s32(f+i);
        v=mul_vec_b40(v,inv_vec,P,mu40(P));
        vst1q_s32(f+i,v);
    }
    for(;i<lim;++i) f[i]=mul_mod_b40(f[i],inv,P,mu40(P));
    std::reverse(f+1,f+lim);
}

/* ------------ 多项式乘法 ------------ */
void poly_mul_simd(int* a,int* b,int* c,int n,int P,int gen=3){
    int l=0; while((1<<l)<2*n) ++l;
    int lim=1<<l;
    std::vector<int> A(lim),B(lim),W(lim);
    std::copy(a,a+n,A.begin());
    std::copy(b,b+n,B.begin());
    calc_powg(W.data(),lim,P,gen);
    DIF(A.data(),l,P,W.data());
    DIF(B.data(),l,P,W.data());
    uint64_t MU=mu40(P);
    for(int i=0;i<lim;++i)
        A[i]=mul_mod_b40(A[i],B[i],P,MU);
    DIT(A.data(),l,P,W.data());
    std::copy(A.begin(),A.begin()+2*n-1,c);
}

/* ------------ 简易测试框架 ------------ */
const int MAXN=300000;
static int a[MAXN],b[MAXN],ab[MAXN];

void fRead(int* a,int* b,int* n,int* p,int id){
    std::ifstream fin("/nttdata/"+std::to_string(id)+".in");
    fin>>*n>>*p;
    for(int i=0;i<*n;++i) fin>>a[i];
    for(int i=0;i<*n;++i) fin>>b[i];
}
void fCheck(int* ab,int n,int id){
    std::ifstream fin("/nttdata/"+std::to_string(id)+".out");
    for(int i=0,x;i<2*n-1;++i){fin>>x;if(x!=ab[i]){std::cout<<"多项式乘法结果错误\n";return;}}
    std::cout<<"多项式乘法结果正确\n";
}

int main(){
    for(int id=0;id<=3;++id){
        int n,p; fRead(a,b,&n,&p,id);
        auto t1=std::chrono::high_resolution_clock::now();
        poly_mul_simd(a,b,ab,n,p);
        auto t2=std::chrono::high_resolution_clock::now();
        fCheck(ab,n,id);
        std::cout<<"latency: n="<<n<<" p="<<p<<" => "
                 <<std::chrono::duration<double,std::milli>(t2-t1).count()<<" ms\n";
    }
    return 0;
}
