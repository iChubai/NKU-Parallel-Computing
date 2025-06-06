#include <vector>
#include <string>
#include <iostream>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <fstream>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

using u32  = uint32_t;
using u64  = uint64_t;
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__GNUC__)
    #if defined(__GNUC__) || defined(__clang__)
        using u128 = __uint128_t;
    #else
        struct u128 { uint64_t lo, hi; };
    #endif
#elif defined(__GNUC__) || defined(__clang__)
using u128 = __uint128_t;
#else
struct u128 { uint64_t lo, hi; }; 
#endif

class Barrett {
public:
    explicit Barrett(u32 m): mod(m) {
        if (m == 0) {
            inv = 0;
            return;
        }
        inv = (static_cast<u128>(1) << 64) / m;
    }

    inline u32 reduce(u64 x) const {
        if (mod == 0) return static_cast<u32>(x);
        u64 q = (static_cast<u128>(x) * inv) >> 64;
        u64 r = x - q * mod;
        if (r >= mod) r -= mod;
        return static_cast<u32>(r);
    }

    inline u32 mul(u32 a, u32 b) const {
        return reduce(static_cast<u64>(a) * b);
    }

    const u32 mod;
public: 
    u64 inv; 
};

static u32 mod_pow(u32 a, u64 e, u32 mod_val) {
    u64 res = 1;
    u64 base = a;
    base %= mod_val;
    while (e > 0) {
        if (e % 2 == 1) res = (static_cast<u128>(res) * base) % mod_val;
        base = (static_cast<u128>(base) * base) % mod_val;
        e /= 2;
    }
    return static_cast<u32>(res);
}

static void bit_reverse_permute(std::vector<u32>& a) {
    int n = a.size();
    if (n == 0) return;
    int lg = 0;
    while ((1 << lg) < n) {
        lg++;
    }
    if ((1 << lg) != n && n != 0) {
        return; 
    }
    if (n==0) return;

    std::vector<int> rev(n);
    rev[0] = 0;
    for (int i = 1; i < n; ++i) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (lg - 1));
    }

    for (int i = 0; i < n; ++i) {
        if (i < rev[i]) {
            std::swap(a[i], a[rev[i]]);
        }
    }
}

void ntt_scalar(std::vector<u32>& a, bool inverse, const Barrett& br, u32 g = 3) {
    int n = a.size();
    if (n == 0 || (n & (n - 1)) != 0) {
        return;
    }

    bit_reverse_permute(a);

    for (int len = 2; len <= n; len <<= 1) {
        int m = len >> 1;
        u32 wn = mod_pow(g, (br.mod - 1) / len, br.mod);
        if (inverse) {
            wn = mod_pow(wn, br.mod - 2, br.mod);
        }
        for (int i = 0; i < n; i += len) {
            u32 w = 1;
            for (int j = 0; j < m; ++j) {
                u32 u = a[i + j];
                u32 v = br.mul(a[i + j + m], w);
                a[i + j]     = (u + v >= br.mod) ? (u + v - br.mod) : (u + v);
                a[i + j + m] = (u >= v) ? (u - v) : (u + br.mod - v);
                w = br.mul(w, wn);
            }
        }
    }

    if (inverse) {
        u32 inv_n = mod_pow(n, br.mod - 2, br.mod);
        for (u32 &x : a) {
            x = br.mul(x, inv_n);
        }
    }
}

void ntt_neon(std::vector<u32>& a, bool inverse, const Barrett& br, u32 g = 3);

bool fRead_poly(std::vector<u32>& vec_a, u32& n_val, u32& p_val, int id) {
    std::string path = "/nttdata/" + std::to_string(id) + ".in";
    std::ifstream fin(path);
    if (!fin) {
        return false;
    }
    int temp_n_int, temp_p_int;
    fin >> temp_n_int >> temp_p_int;
    n_val = static_cast<u32>(temp_n_int);
    p_val = static_cast<u32>(temp_p_int);

    if (n_val == 0 || (n_val & (n_val - 1)) != 0) {
        return false;
    }

    vec_a.resize(n_val);
    for (u32 i = 0; i < n_val; ++i) {
        int val;
        if (!(fin >> val)) {
            return false;
        }
        vec_a[i] = (static_cast<long long>(val) % p_val + p_val) % p_val;
    }
    fin.close();
    return true;
}

int main() {
    for (int id = 0; id <= 3; ++id) {
        u32 n_val, p_val;
        std::vector<u32> poly_orig;

        if (!fRead_poly(poly_orig, n_val, p_val, id)) {
            continue;
        }

        Barrett br(p_val);

        std::vector<u32> poly_neon = poly_orig;
        std::vector<u32> poly_scalar = poly_orig;

        ntt_scalar(poly_scalar, false, br);
        ntt_scalar(poly_scalar, true, br);

    #ifdef __ARM_NEON
        ntt_neon(poly_neon, false, br);
        ntt_neon(poly_neon, true, br);
    #endif
    }
    return 0;
}

#ifdef __ARM_NEON

inline uint32x4_t barrett_mul_u32x4(uint32x4_t val_a, uint32x4_t val_b, u32 br_mod, u64 br_inv, const Barrett& br_obj_ref) {
    uint32x2_t a_lo = vget_low_u32(val_a);
    uint32x2_t a_hi = vget_high_u32(val_a);
    uint32x2_t b_lo = vget_low_u32(val_b);
    uint32x2_t b_hi = vget_high_u32(val_b);

    uint64x2_t x_lo = vmull_u32(a_lo, b_lo); 
    uint64x2_t x_hi = vmull_u32(a_hi, b_hi); 

    #if defined(__aarch64__) && defined(__ARM_FEATURE_SHA512) 
        uint64x2_t inv_vec_duplicated = vdupq_n_u64(br_inv);
        uint64x2_t q_lo = vmulh_u64(x_lo, inv_vec_duplicated);
        uint64x2_t q_hi = vmulh_u64(x_hi, inv_vec_duplicated);

        uint64x2_t mod_u64_duplicated = vdupq_n_u64(br_mod);
        uint64x2_t q_mod_lo = vmulq_u64(q_lo, mod_u64_duplicated);
        uint64x2_t q_mod_hi = vmulq_u64(q_hi, mod_u64_duplicated);

        uint64x2_t r_lo = vsubq_u64(x_lo, q_mod_lo);
        uint64x2_t r_hi = vsubq_u64(x_hi, q_mod_hi);

        uint64x2_t mask_lo = vcgeq_u64(r_lo, mod_u64_duplicated);
        r_lo = vbslq_u64(mask_lo, vsubq_u64(r_lo, mod_u64_duplicated), r_lo);
        
        uint64x2_t mask_hi = vcgeq_u64(r_hi, mod_u64_duplicated);
        r_hi = vbslq_u64(mask_hi, vsubq_u64(r_hi, mod_u64_duplicated), r_hi);

        uint32x2_t res_narrow_lo = vmovn_u64(r_lo);
        uint32x2_t res_narrow_hi = vmovn_u64(r_hi);
        return vcombine_u32(res_narrow_lo, res_narrow_hi);
    #else 
        u32 results[4];
        u32 current_a[4], current_b[4];
        vst1q_u32(current_a, val_a);
        vst1q_u32(current_b, val_b);
        for(int k=0; k<4; ++k) {
            results[k] = br_obj_ref.mul(current_a[k], current_b[k]);
        }
        return vld1q_u32(results);
    #endif
}

void ntt_neon(std::vector<u32>& a, bool inverse, const Barrett& br_obj, u32 g) {
    int n = a.size();
    if (n == 0 || (n & (n - 1)) != 0) { 
        return;
    }
    if (n < 4 && n > 0) { 
        ntt_scalar(a, inverse, br_obj, g);
        return;
    }

    bit_reverse_permute(a);

    u32 br_mod = br_obj.mod;
    u64 br_inv = br_obj.inv; 
    uint32x4_t mod_vec = vdupq_n_u32(br_mod);

    for (int len = 2; len <= n; len <<= 1) {
        int m = len >> 1;
        u32 wn_scalar = mod_pow(g, (br_mod - 1) / len, br_mod);
        if (inverse) {
            wn_scalar = mod_pow(wn_scalar, br_mod - 2, br_mod);
        }

        for (int i = 0; i < n; i += len) {
            u32 w_scalar = 1;
            int j = 0;
            for (; j + 3 < m; j += 4) { 
                uint32x4_t u_vec = vld1q_u32(&a[i + j]);
                uint32x4_t v_input_vec = vld1q_u32(&a[i + j + m]);

                uint32_t w_terms_scalar[4];
                w_terms_scalar[0] = w_scalar;
                w_terms_scalar[1] = br_obj.mul(w_terms_scalar[0], wn_scalar);
                w_terms_scalar[2] = br_obj.mul(w_terms_scalar[1], wn_scalar);
                w_terms_scalar[3] = br_obj.mul(w_terms_scalar[2], wn_scalar);
                uint32x4_t w_vec = vld1q_u32(w_terms_scalar);
                
                uint32x4_t v_vec = barrett_mul_u32x4(v_input_vec, w_vec, br_mod, br_inv, br_obj);
                
                uint32x4_t u_plus_v = vaddq_u32(u_vec, v_vec);
                uint32x4_t u_minus_v = vsubq_u32(u_vec, v_vec);

                uint32x4_t mask_add = vcgeq_u32(u_plus_v, mod_vec);
                u_plus_v = vbslq_u32(mask_add, vsubq_u32(u_plus_v, mod_vec), u_plus_v);
                
                uint32x4_t mask_sub_neg = vcltq_s32(vreinterpretq_s32_u32(u_minus_v), vdupq_n_s32(0)); 
                u_minus_v = vbslq_u32(mask_sub_neg, vaddq_u32(u_minus_v, mod_vec), u_minus_v);

                vst1q_u32(&a[i + j], u_plus_v);
                vst1q_u32(&a[i + j + m], u_minus_v);

                w_scalar = br_obj.mul(w_terms_scalar[3], wn_scalar);
            }
            for (; j < m; ++j) {
                u32 u = a[i + j];
                u32 v = br_obj.mul(a[i + j + m], w_scalar);
                a[i + j]     = (u + v >= br_mod) ? (u + v - br_mod) : (u + v);
                a[i + j + m] = (u >= v) ? (u - v) : (u + br_mod - v);
                w_scalar = br_obj.mul(w_scalar, wn_scalar);
            }
        }
    }

    if (inverse) {
        u32 inv_n_scalar = mod_pow(n, br_mod - 2, br_mod);
        for (u32 &x : a) { 
            x = br_obj.mul(x, inv_n_scalar);
        }
    }
}
#endif
