// ===========================================
// 文件名: main_DIF_DIT_neon.cpp
// 描述  : 使用 NEON SIMD 优化 DIF / DIT‑NTT 的多项式乘法
// ===========================================

#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using ll = long long;

void fRead(int *a, int *b, int *n, int *p, int input_id) {
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

void fCheck(int *ab, int n, int input_id) {
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
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

void fWrite(int *ab, int n, int input_id) {
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
// ===============================  工具宏  =========================================
#define VEC_WIDTH 4  // int32x4_t

inline int32x4_t mod_add_vec(int32x4_t a, int32x4_t b, int32x4_t P) {
    int32x4_t s  = vaddq_s32(a, b);
    int32x4_t ge = vreinterpretq_s32_u32(vcgeq_s32(s, P));
    return vsubq_s32(s, vandq_s32(ge, P));
}

inline int32x4_t mod_sub_vec(int32x4_t a, int32x4_t b, int32x4_t P) {
    int32x4_t d  = vsubq_s32(a, b);
    int32x4_t lt = vreinterpretq_s32_u32(vcltq_s32(d, vdupq_n_s32(0)));
    return vaddq_s32(d, vandq_s32(lt, P));
}

static inline int mul_mod(int a, int b, int P) {
    return (int)((1LL * a * b) % P);
}
static inline int32x4_t mod_mul_vec(int32x4_t a, int32x4_t b,
                                    int32x4_t P, uint32x4_t mu) {
    int64x2_t prod_lo = vmull_s32(vget_low_s32(a),  vget_low_s32(b));
    int64x2_t prod_hi = vmull_s32(vget_high_s32(a), vget_high_s32(b));

    uint32x2_t q_lo = vreinterpret_u32_s32(vshrn_n_s64(prod_lo, 32));
    uint32x2_t q_hi = vreinterpret_u32_s32(vshrn_n_s64(prod_hi, 32));
    uint32x4_t q    = vcombine_u32(q_lo, q_hi);

    uint64x2_t t_lo64 = vmull_u32(vget_low_u32(q),  vget_low_u32(mu));
    uint64x2_t t_hi64 = vmull_u32(vget_high_u32(q), vget_high_u32(mu));
    uint32x2_t t_lo   = vshrn_n_u64(t_lo64, 32);
    uint32x2_t t_hi   = vshrn_n_u64(t_hi64, 32);
    int32x4_t  t      = vreinterpretq_s32_u32(vcombine_u32(t_lo, t_hi));

    int64x2_t tP_lo = vmull_s32(vget_low_s32(t),  vget_low_s32(P));
    int64x2_t tP_hi = vmull_s32(vget_high_s32(t), vget_high_s32(P));
    int64x2_t r_lo  = vsubq_s64(prod_lo, tP_lo);
    int64x2_t r_hi  = vsubq_s64(prod_hi, tP_hi);
    int32x2_t r2_lo = vmovn_s64(r_lo);
    int32x2_t r2_hi = vmovn_s64(r_hi);
    int32x4_t r2    = vcombine_s32(r2_lo, r2_hi);
    uint32x4_t ge   = vcgeq_s32(r2, P);
    int32x4_t mask  = vreinterpretq_s32_u32(ge);
    return vsubq_s32(r2, vandq_s32(mask, P));
}
inline int32x4_t mul_mod_vec(int32x4_t v, int w, int P) {
    int32x4_t res;
    for (int lane = 0; lane < VEC_WIDTH; ++lane) {
        int val = vgetq_lane_s32(v, lane);
        val     = mul_mod(val, w, P);
        res     = vsetq_lane_s32(val, res, lane);
    }
    return res;
}

// =======================  生成原根幂表（与原版相同）  ==============================
static inline ll fpow(ll a, ll b, int P) {
    ll r = 1;
    for (a %= P; b; b >>= 1, a = (a * a) % P)
        if (b & 1) r = (r * a) % P;
    return r;
}

void calc_powg(int *w, int G, int P, int gen) {
    w[0] = 1;
    const int g = fpow(gen, (P - 1) / G, P);
    for (int t = 0; (1 << (t + 1)) < G; ++t) {
        ll f          = w[1 << t] = fpow(g, G >> (t + 2), P);
        const int blk = 1 << t;
        for (int x = blk; x < (blk << 1); ++x) w[x] = (ll)f * w[x - blk] % P;
    }
}

// =============================  DIF/DIT (NEON)  ==================================
void DIF_neon(int *f, int l, int P, const int *w) {
    const int lim = 1 << l;
    const int32x4_t Pvec = vdupq_n_s32(P);

    for (int len = lim; len > 1; len >>= 1) {
        const int half = len >> 1;
        for (int st = 0, t = 0; st < lim; st += len, ++t) {
            const int wt = w[t];  // 同一块共用同一 w
            int i = st;
            // 4‑way SIMD
            for (; i + VEC_WIDTH - 1 < st + half; i += VEC_WIDTH) {
                int32x4_t g = vld1q_s32(f + i);
                int32x4_t h = vld1q_s32(f + i + half);
                h            = mul_mod_vec(h, wt, P);  // 标量乘模 + 向量装载

                int32x4_t sum = mod_add_vec(g, h, Pvec);
                int32x4_t dif = mod_sub_vec(g, h, Pvec);

                vst1q_s32(f + i, sum);
                vst1q_s32(f + i + half, dif);
            }
            for (; i < st + half; ++i) {
                ll g = f[i];
                ll h = 1LL * f[i + half] * wt % P;
                f[i]         = (g + h) % P;
                f[i + half]  = (P + g - h) % P;
            }
        }
    }
}

void DIT_neon(int *f, int l, int P, const int *w) {
    const int lim = 1 << l;
    const int32x4_t Pvec = vdupq_n_s32(P);

    for (int len = 2; len <= lim; len <<= 1) {
        const int half = len >> 1;
        for (int st = 0, t = 0; st < lim; st += len, ++t) {
            const int wt = w[t];
            int i = st;
            for (; i + VEC_WIDTH - 1 < st + half; i += VEC_WIDTH) {
                int32x4_t g = vld1q_s32(f + i);
                int32x4_t h = vld1q_s32(f + i + half);

                int32x4_t sum = mod_add_vec(g, h, Pvec);
                int32x4_t dif = mod_sub_vec(g, h, Pvec);
                dif            = mul_mod_vec(dif, wt, P);

                vst1q_s32(f + i, sum);
                vst1q_s32(f + i + half, dif);
            }
            // scalar tail
            for (; i < st + half; ++i) {
                ll g = f[i];
                ll h = f[i + half];
                f[i]         = (g + h) % P;
                f[i + half]  = (P + g - h) * 1LL * wt % P;
            }
        }
    }
    const ll invl = fpow(lim, P - 2, P);
    int32x4_t inv_vec = vdupq_n_s32((int)invl);
    int i = 0;
    for (; i + VEC_WIDTH - 1 < lim; i += VEC_WIDTH) {
        int32x4_t v = vld1q_s32(f + i);
        v           = mul_mod_vec(v, (int)invl, P);
        vst1q_s32(f + i, v);
    }
    for (; i < lim; ++i) f[i] = (ll)f[i] * invl % P;

    std::reverse(f + 1, f + lim);
}

// ==========================  多项式乘法（入口）  ================================
void poly_multiply_simd(int *a, int *b, int *ab, int n, int P, int gen = 3) {
    std::fill(ab, ab + 2 * n - 1, 0);

    int l  = 0;
    while ((1 << l) < 2 * n) ++l;
    const int lim = 1 << l;

    std::unique_ptr<int[]> A(new int[lim]());
    std::unique_ptr<int[]> B(new int[lim]());
    std::unique_ptr<int[]> W(new int[lim]());

    std::copy(a, a + n, A.get());
    std::copy(b, b + n, B.get());

    calc_powg(W.get(), lim, P, gen);

    DIF_neon(A.get(), l, P, W.get());
    DIF_neon(B.get(), l, P, W.get());

    // 点乘仍使用标量（乘模开销相对较小）
    for (int i = 0; i < lim; ++i) A[i] = 1LL * A[i] * B[i] % P;

    DIT_neon(A.get(), l, P, W.get());

    std::copy(A.get(), A.get() + 2 * n - 1, ab);
}

// ======================  原 main 仅替换调用接口  ================================
const int MAXN = 300000;
static int a[MAXN], b[MAXN], ab[MAXN];

void fRead(int *a, int *b, int *n, int *p, int input_id);  // implementation unchanged
void fCheck(int *ab, int n, int input_id);
void fWrite(int *ab, int n, int input_id);

int main() {
    int test_begin = 0, test_end = 3;
    for (int i = test_begin; i <= test_end; ++i) {
        int n_, p_;
        fRead(a, b, &n_, &p_, i);

        auto tic = std::chrono::high_resolution_clock::now();
        poly_multiply_simd(a, b, ab, n_, p_);
        auto toc = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> dt = toc - tic;
        fCheck(ab, n_, i);
        std::cout << "latency: n=" << n_ << " p=" << p_ << " => " << dt.count() << " ms\n";
        fWrite(ab, n_, i);
    }
    return 0;
}
