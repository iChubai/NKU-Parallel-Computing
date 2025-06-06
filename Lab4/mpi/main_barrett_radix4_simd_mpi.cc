#include <bits/stdc++.h>
#include <mpi.h>
#include <iomanip>
#include <chrono>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

using namespace std;

using u32  = unsigned int;
using u64  = unsigned long long;
#if defined(_MSC_VER) && !defined(__clang__)
using u128 = unsigned __int128;
#else
using u128 = __uint128_t;
#endif

void fRead(int *a, int *b, int *n, int *p, int input_id) {
    string path = "/nttdata/" + to_string(input_id) + ".in";
    ifstream fin(path);
    if(!fin) { 
        cerr << "无法打开输入文件: " << path << '\n'; 
        MPI_Abort(MPI_COMM_WORLD, 1); 
    }
    fin >> *n >> *p;
    for (int i = 0; i < *n; ++i) fin >> a[i];
    for (int i = 0; i < *n; ++i) fin >> b[i];
}

void fCheck(int *ab, int n, int input_id) {
    string path = "/nttdata/" + to_string(input_id) + ".out";
    ifstream fin(path);
    if(!fin) { 
        cerr << "无法打开输出文件: " << path << '\n'; 
        MPI_Abort(MPI_COMM_WORLD, 1); 
    }
    for (int i = 0; i < 2 * n - 1; ++i) {
        int x; 
        fin >> x;
        if (x != ab[i]) { 
            cout << "多项式乘法结果错误 (id="<<input_id<<")\n"; 
            return; 
        }
    }
    cout << "多项式乘法结果正确 (id="<<input_id<<")\n";
}

void fWrite(int *ab, int n, int input_id) {
    string path = "files/" + to_string(input_id) + ".out";
    ofstream fout(path);
    for (int i = 0; i < 2 * n - 1; ++i) fout << ab[i] << '\n';
}

class Barrett {
public:
    explicit Barrett(u32 m): mod(m) {
        inv = (static_cast<u128>(1) << 64) / m;
    }
    
    inline u32 reduce(u64 x) const {
        u64 q = (static_cast<u128>(x) * inv) >> 64;
        u64 r = x - q * mod;
        if (r >= mod) r -= mod;
        return static_cast<u32>(r);
    }
    
    inline u32 mul(u32 a, u32 b) const {
        return reduce(static_cast<u64>(a) * b);
    }
    
    inline u32 add(u32 a, u32 b) const {
        u32 s = a + b;
        return s >= mod ? s - mod : s;
    }
    
    inline u32 sub(u32 a, u32 b) const {
        return a >= b ? a - b : a + mod - b;
    }
    
    const u32 mod;
    u64 inv;
};

static u32 mod_pow(u32 a, u64 e, u32 mod) {
    u64 res = 1, base = a;
    while (e) {
        if (e & 1) res = res * base % mod;
        base = base * base % mod;
        e >>= 1;
    }
    return static_cast<u32>(res);
}

static void bit_reverse(vector<int>& rev, int n) {
    int lg = __builtin_ctz(n);
    rev.resize(n);
    for (int i = 0; i < n; ++i) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (lg - 1));
    }
}

static void digit_reverse_4(vector<u32>& a) {
    int n = a.size();
    int pairs = __builtin_ctz(n) >> 1;
    for(int i = 0; i < n; ++i) {
        int rev = 0, t = i;
        for(int j = 0; j < pairs; ++j) { 
            rev = (rev << 2) | (t & 3); 
            t >>= 2; 
        }
        if(i < rev) swap(a[i], a[rev]);
    }
}

struct MPIContext {
    int rank, size;
    MPI_Comm comm;
};

#ifdef __ARM_NEON
inline uint32x4_t barrett_mul_neon(uint32x4_t val_a, uint32x4_t val_b, const Barrett& br) {
    uint32x2_t a_lo = vget_low_u32(val_a);
    uint32x2_t a_hi = vget_high_u32(val_a);
    uint32x2_t b_lo = vget_low_u32(val_b);
    uint32x2_t b_hi = vget_high_u32(val_b);
    
    uint64x2_t x_lo = vmull_u32(a_lo, b_lo);
    uint64x2_t x_hi = vmull_u32(a_hi, b_hi);
    
    #if defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
    uint64x2_t inv_vec = vdupq_n_u64(br.inv);
    uint64x2_t mod_vec = vdupq_n_u64(br.mod);

    uint64_t x_vals[4];
    uint64_t inv_val = br.inv, mod_val = br.mod;
    vst1q_u64(x_vals, x_lo);
    vst1q_u64(x_vals + 2, x_hi);

    uint64_t q_vals[4], r_vals[4];
    for(int i = 0; i < 4; ++i) {
        u128 temp = (u128)x_vals[i] * inv_val;
        q_vals[i] = temp >> 64;
        r_vals[i] = x_vals[i] - q_vals[i] * mod_val;
        if(r_vals[i] >= mod_val) r_vals[i] -= mod_val;
    }

    uint32_t results[4];
    for(int i = 0; i < 4; ++i) {
        results[i] = static_cast<uint32_t>(r_vals[i]);
    }
    return vld1q_u32(results);

    #else
    uint32_t results[4];
    uint32_t a_vals[4], b_vals[4];
    vst1q_u32(a_vals, val_a);
    vst1q_u32(b_vals, val_b);
    for (int i = 0; i < 4; ++i) {
        results[i] = br.mul(a_vals[i], b_vals[i]);
    }
    return vld1q_u32(results);
    #endif
}

inline uint32x4_t mod_add_neon(uint32x4_t a, uint32x4_t b, uint32x4_t mod_vec) {
    uint32x4_t sum = vaddq_u32(a, b);
    uint32x4_t mask = vcgeq_u32(sum, mod_vec);
    return vbslq_u32(mask, vsubq_u32(sum, mod_vec), sum);
}

inline uint32x4_t mod_sub_neon(uint32x4_t a, uint32x4_t b, uint32x4_t mod_vec) {
    uint32x4_t diff = vsubq_u32(a, b);
    uint32x4_t mask = vcltq_u32(a, b);
    return vbslq_u32(mask, vaddq_u32(diff, mod_vec), diff);
}
#endif

static void radix2_layer_serial(vector<u32>& a, bool inv, const Barrett& br, u32 g = 3) {
    int n = a.size();
    int h = n >> 1;
    u32 wn = mod_pow(g, (br.mod - 1) / n, br.mod);
    if(inv) wn = mod_pow(wn, br.mod - 2, br.mod);
    u32 w = 1;

    for(int j = 0; j < h; ++j) {
        u32 u = a[j];
        u32 v = br.mul(a[j + h], w);
        a[j] = br.add(u, v);
        a[j + h] = br.sub(u, v);
        w = br.mul(w, wn);
    }
}

#ifdef __ARM_NEON
inline void radix4_butterfly_neon(uint32x4_t& a0, uint32x4_t& a1, uint32x4_t& a2, uint32x4_t& a3,
                                  uint32x4_t w1, uint32x4_t w2, uint32x4_t w3, uint32x4_t J_factor,
                                  const Barrett& br, uint32x4_t mod_vec) {
    uint32x4_t A = a0;
    uint32x4_t B = barrett_mul_neon(a1, w1, br);
    uint32x4_t C = barrett_mul_neon(a2, w2, br);
    uint32x4_t D = barrett_mul_neon(a3, w3, br);

    uint32x4_t T0 = mod_add_neon(A, C, mod_vec);
    uint32x4_t T1 = mod_sub_neon(A, C, mod_vec);
    uint32x4_t T2 = mod_add_neon(B, D, mod_vec);
    uint32x4_t T3 = barrett_mul_neon(mod_sub_neon(B, D, mod_vec), J_factor, br);

    a0 = mod_add_neon(T0, T2, mod_vec);
    a1 = mod_add_neon(T1, T3, mod_vec);
    a2 = mod_sub_neon(T0, T2, mod_vec);
    a3 = mod_sub_neon(T1, T3, mod_vec);
}
#endif

void ntt_radix4_simd_parallel(vector<u32>& a, bool inverse, const Barrett& br,
                              const MPIContext& ctx, u32 g = 3) {
    int n = a.size();
    bool odd = __builtin_ctz(n) & 1;

    if(ctx.rank == 0) {
        if(odd) {
            vector<int> rev;
            bit_reverse(rev, n);
            for(int i = 0; i < n; ++i) {
                if(i < rev[i]) swap(a[i], a[rev[i]]);
            }
            radix2_layer_serial(a, inverse, br, g);
        } else {
            digit_reverse_4(a);
        }
    }
    MPI_Bcast(a.data(), n, MPI_UNSIGNED, 0, ctx.comm);

#ifdef __ARM_NEON
    uint32x4_t mod_vec = vdupq_n_u32(br.mod);
#endif

    int start_len = odd ? 8 : 4;
    for(int len = start_len; len <= n; len <<= 2) {
        int m = len >> 2;
        if(m <= 0) continue;

        u32 wn_base = mod_pow(g, (br.mod - 1) / len, br.mod);
        if(inverse) wn_base = mod_pow(wn_base, br.mod - 2, br.mod);
        u32 J_factor = mod_pow(wn_base, m, br.mod);

        vector<u32> w_pow_j(m);
        w_pow_j[0] = 1;
        for(int j_idx = 1; j_idx < m; ++j_idx) {
            w_pow_j[j_idx] = br.mul(w_pow_j[j_idx - 1], wn_base);
        }

        int totalRows = n / len;
        if(totalRows <= 0) continue;

        int base_rows_per_proc = totalRows / ctx.size;
        int rem_rows = totalRows % ctx.size;
        int myRows = base_rows_per_proc + (ctx.rank < rem_rows ? 1 : 0);
        int rowStartOffset = ctx.rank * base_rows_per_proc + min(ctx.rank, rem_rows);

        for(int r_idx = 0; r_idx < myRows; ++r_idx) {
            int blk_offset = (rowStartOffset + r_idx) * len;
            if(blk_offset + len > n) break;

            int j_group_idx = 0;

#ifdef __ARM_NEON
            uint32x4_t J_factor_vec = vdupq_n_u32(J_factor);

            for(; j_group_idx + 3 < m; j_group_idx += 4) {
                if(blk_offset + j_group_idx + 3 * m >= n) break;

                uint32x4_t a0_vec = vld1q_u32(&a[blk_offset + j_group_idx]);
                uint32x4_t a1_vec = vld1q_u32(&a[blk_offset + j_group_idx + m]);
                uint32x4_t a2_vec = vld1q_u32(&a[blk_offset + j_group_idx + 2 * m]);
                uint32x4_t a3_vec = vld1q_u32(&a[blk_offset + j_group_idx + 3 * m]);

                u32 w_terms[4];
                for(int k = 0; k < 4; ++k) {
                    if(j_group_idx + k < m) {
                        w_terms[k] = w_pow_j[j_group_idx + k];
                    } else {
                        w_terms[k] = 1;
                    }
                }
                uint32x4_t w1_vec = vld1q_u32(w_terms);

                u32 w2_terms[4];
                for(int k = 0; k < 4; ++k) {
                    w2_terms[k] = br.mul(w_terms[k], w_terms[k]);
                }
                uint32x4_t w2_vec = vld1q_u32(w2_terms);

                u32 w3_terms[4];
                for(int k = 0; k < 4; ++k) {
                    w3_terms[k] = br.mul(w2_terms[k], w_terms[k]);
                }
                uint32x4_t w3_vec = vld1q_u32(w3_terms);

                radix4_butterfly_neon(a0_vec, a1_vec, a2_vec, a3_vec,
                                     w1_vec, w2_vec, w3_vec, J_factor_vec,
                                     br, mod_vec);

                vst1q_u32(&a[blk_offset + j_group_idx], a0_vec);
                vst1q_u32(&a[blk_offset + j_group_idx + m], a1_vec);
                vst1q_u32(&a[blk_offset + j_group_idx + 2 * m], a2_vec);
                vst1q_u32(&a[blk_offset + j_group_idx + 3 * m], a3_vec);
            }
#endif

            for(; j_group_idx < m; ++j_group_idx) {
                if(blk_offset + j_group_idx + 3 * m >= n) break;

                u32 current_w_j = w_pow_j[j_group_idx];
                u32 current_w_2j = br.mul(current_w_j, current_w_j);
                u32 current_w_3j = br.mul(current_w_2j, current_w_j);

                u32 termA_in = a[blk_offset + j_group_idx];
                u32 termB_in = a[blk_offset + j_group_idx + m];
                u32 termC_in = a[blk_offset + j_group_idx + 2 * m];
                u32 termD_in = a[blk_offset + j_group_idx + 3 * m];

                u32 A_tw = termA_in;
                u32 B_tw = br.mul(termB_in, current_w_j);
                u32 C_tw = br.mul(termC_in, current_w_2j);
                u32 D_tw = br.mul(termD_in, current_w_3j);

                u32 T0 = br.add(A_tw, C_tw);
                u32 T1 = br.sub(A_tw, C_tw);
                u32 T2 = br.add(B_tw, D_tw);
                u32 T3 = br.mul(br.sub(B_tw, D_tw), J_factor);

                a[blk_offset + j_group_idx]        = br.add(T0, T2);
                a[blk_offset + j_group_idx + m]    = br.add(T1, T3);
                a[blk_offset + j_group_idx + 2*m]  = br.sub(T0, T2);
                a[blk_offset + j_group_idx + 3*m]  = br.sub(T1, T3);
            }
        }

        vector<int> recvcounts(ctx.size), displs(ctx.size);
        for(int r_proc = 0; r_proc < ctx.size; ++r_proc) {
            int rows_for_r_proc = base_rows_per_proc + (r_proc < rem_rows ? 1 : 0);
            recvcounts[r_proc] = rows_for_r_proc * len;
            displs[r_proc] = (r_proc * base_rows_per_proc + min(r_proc, rem_rows)) * len;
        }

        int sendcnt_local = myRows * len;
        if(sendcnt_local > 0 && rowStartOffset * len < n) {
            int actual_copy_size = min(sendcnt_local, n - rowStartOffset * len);
            vector<u32> sendbuf_local_storage(sendcnt_local, 0);

            if(actual_copy_size > 0) {
                memcpy(sendbuf_local_storage.data(), a.data() + rowStartOffset * len,
                       sizeof(u32) * actual_copy_size);
            }

            MPI_Allgatherv(sendbuf_local_storage.data(), sendcnt_local, MPI_UNSIGNED,
                           a.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                           ctx.comm);
        } else {
            MPI_Allgatherv(nullptr, 0, MPI_UNSIGNED,
                           a.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                           ctx.comm);
        }
    }

    if(inverse) {
        if(ctx.rank == 0) {
            u32 invN = mod_pow(n, br.mod - 2, br.mod);
#ifdef __ARM_NEON
            uint32x4_t inv_n_vec = vdupq_n_u32(invN);
            int i = 0;
            for(; i + 3 < n; i += 4) {
                uint32x4_t x_vec = vld1q_u32(&a[i]);
                uint32x4_t result = barrett_mul_neon(x_vec, inv_n_vec, br);
                vst1q_u32(&a[i], result);
            }
            for(; i < n; ++i) {
                a[i] = br.mul(a[i], invN);
            }
#else
            for(u32& x : a) x = br.mul(x, invN);
#endif
        }
        MPI_Bcast(a.data(), n, MPI_UNSIGNED, 0, ctx.comm);
    }
}

void ntt_radix4_scalar_parallel(vector<u32>& a, bool inverse, const Barrett& br,
                                const MPIContext& ctx, u32 g = 3) {
    int n = a.size();
    bool odd = __builtin_ctz(n) & 1;

    if(ctx.rank == 0) {
        if(odd) {
            vector<int> rev;
            bit_reverse(rev, n);
            for(int i = 0; i < n; ++i) {
                if(i < rev[i]) swap(a[i], a[rev[i]]);
            }
            radix2_layer_serial(a, inverse, br, g);
        } else {
            digit_reverse_4(a);
        }
    }
    MPI_Bcast(a.data(), n, MPI_UNSIGNED, 0, ctx.comm);

    int start_len = odd ? 8 : 4;
    for(int len = start_len; len <= n; len <<= 2) {
        int m = len >> 2;
        if(m <= 0) continue;

        u32 wn_base = mod_pow(g, (br.mod - 1) / len, br.mod);
        if(inverse) wn_base = mod_pow(wn_base, br.mod - 2, br.mod);
        u32 J_factor = mod_pow(wn_base, m, br.mod);

        vector<u32> w_pow_j(m);
        w_pow_j[0] = 1;
        for(int j_idx = 1; j_idx < m; ++j_idx) {
            w_pow_j[j_idx] = br.mul(w_pow_j[j_idx - 1], wn_base);
        }

        int totalRows = n / len;
        if(totalRows <= 0) continue;

        int base_rows_per_proc = totalRows / ctx.size;
        int rem_rows = totalRows % ctx.size;
        int myRows = base_rows_per_proc + (ctx.rank < rem_rows ? 1 : 0);
        int rowStartOffset = ctx.rank * base_rows_per_proc + min(ctx.rank, rem_rows);

        for(int r_idx = 0; r_idx < myRows; ++r_idx) {
            int blk_offset = (rowStartOffset + r_idx) * len;
            if(blk_offset + len > n) break;

            for(int j_group_idx = 0; j_group_idx < m; ++j_group_idx) {
                if(blk_offset + j_group_idx + 3 * m >= n) break;

                u32 current_w_j = w_pow_j[j_group_idx];
                u32 current_w_2j = br.mul(current_w_j, current_w_j);
                u32 current_w_3j = br.mul(current_w_2j, current_w_j);

                u32 termA_in = a[blk_offset + j_group_idx];
                u32 termB_in = a[blk_offset + j_group_idx + m];
                u32 termC_in = a[blk_offset + j_group_idx + 2 * m];
                u32 termD_in = a[blk_offset + j_group_idx + 3 * m];

                u32 A_tw = termA_in;
                u32 B_tw = br.mul(termB_in, current_w_j);
                u32 C_tw = br.mul(termC_in, current_w_2j);
                u32 D_tw = br.mul(termD_in, current_w_3j);

                u32 T0 = br.add(A_tw, C_tw);
                u32 T1 = br.sub(A_tw, C_tw);
                u32 T2 = br.add(B_tw, D_tw);
                u32 T3 = br.mul(br.sub(B_tw, D_tw), J_factor);

                a[blk_offset + j_group_idx]        = br.add(T0, T2);
                a[blk_offset + j_group_idx + m]    = br.add(T1, T3);
                a[blk_offset + j_group_idx + 2*m]  = br.sub(T0, T2);
                a[blk_offset + j_group_idx + 3*m]  = br.sub(T1, T3);
            }
        }

        vector<int> recvcounts(ctx.size), displs(ctx.size);
        for(int r_proc = 0; r_proc < ctx.size; ++r_proc) {
            int rows_for_r_proc = base_rows_per_proc + (r_proc < rem_rows ? 1 : 0);
            recvcounts[r_proc] = rows_for_r_proc * len;
            displs[r_proc] = (r_proc * base_rows_per_proc + min(r_proc, rem_rows)) * len;
        }

        int sendcnt_local = myRows * len;
        if(sendcnt_local > 0 && rowStartOffset * len < n) {
            int actual_copy_size = min(sendcnt_local, n - rowStartOffset * len);
            vector<u32> sendbuf_local_storage(sendcnt_local, 0);

            if(actual_copy_size > 0) {
                memcpy(sendbuf_local_storage.data(), a.data() + rowStartOffset * len,
                       sizeof(u32) * actual_copy_size);
            }

            MPI_Allgatherv(sendbuf_local_storage.data(), sendcnt_local, MPI_UNSIGNED,
                           a.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                           ctx.comm);
        } else {
            MPI_Allgatherv(nullptr, 0, MPI_UNSIGNED,
                           a.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                           ctx.comm);
        }
    }

    if(inverse) {
        if(ctx.rank == 0) {
            u32 invN = mod_pow(n, br.mod - 2, br.mod);
            for(u32& x : a) x = br.mul(x, invN);
        }
        MPI_Bcast(a.data(), n, MPI_UNSIGNED, 0, ctx.comm);
    }
}

void poly_multiply_radix4_simd_parallel(const int* a, const int* b, int* ab, int n, int p,
                                        const MPIContext& ctx) {
    Barrett br(p);
    int lim = 1;
    while(lim < 2*n) lim <<= 1;

    vector<u32> A(lim, 0), B(lim, 0);

    if(ctx.rank == 0) {
        for(int i = 0; i < n; ++i) {
            A[i] = ((a[i] % p) + p) % p;
            B[i] = ((b[i] % p) + p) % p;
        }
    }

    MPI_Bcast(A.data(), lim, MPI_UNSIGNED, 0, ctx.comm);
    MPI_Bcast(B.data(), lim, MPI_UNSIGNED, 0, ctx.comm);

    ntt_radix4_simd_parallel(A, false, br, ctx);
    ntt_radix4_simd_parallel(B, false, br, ctx);

#ifdef __ARM_NEON
    int i = 0;
    for(; i + 3 < lim; i += 4) {
        uint32x4_t a_vec = vld1q_u32(&A[i]);
        uint32x4_t b_vec = vld1q_u32(&B[i]);
        uint32x4_t result = barrett_mul_neon(a_vec, b_vec, br);
        vst1q_u32(&A[i], result);
    }
    for(; i < lim; ++i) {
        A[i] = br.mul(A[i], B[i]);
    }
#else
    for(int i = 0; i < lim; ++i) {
        A[i] = br.mul(A[i], B[i]);
    }
#endif

    ntt_radix4_simd_parallel(A, true, br, ctx);

    if(ctx.rank == 0) {
        for(int i = 0; i < 2*n-1; ++i) {
            ab[i] = static_cast<int>(A[i]);
        }
    }
}

void poly_multiply_radix4_scalar_parallel(const int* a, const int* b, int* ab, int n, int p,
                                         const MPIContext& ctx) {
    Barrett br(p);
    int lim = 1;
    while(lim < 2*n) lim <<= 1;

    vector<u32> A(lim, 0), B(lim, 0);

    if(ctx.rank == 0) {
        for(int i = 0; i < n; ++i) {
            A[i] = ((a[i] % p) + p) % p;
            B[i] = ((b[i] % p) + p) % p;
        }
    }

    MPI_Bcast(A.data(), lim, MPI_UNSIGNED, 0, ctx.comm);
    MPI_Bcast(B.data(), lim, MPI_UNSIGNED, 0, ctx.comm);

    ntt_radix4_scalar_parallel(A, false, br, ctx);
    ntt_radix4_scalar_parallel(B, false, br, ctx);

    for(int i = 0; i < lim; ++i) {
        A[i] = br.mul(A[i], B[i]);
    }

    ntt_radix4_scalar_parallel(A, true, br, ctx);

    if(ctx.rank == 0) {
        for(int i = 0; i < 2*n-1; ++i) {
            ab[i] = static_cast<int>(A[i]);
        }
    }
}

static int a[300000], b[300000], ab[600000];

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    MPIContext ctx;
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);
    ctx.comm = MPI_COMM_WORLD;

    const int first = 0;
    const int last  = 3;

    if(ctx.rank == 0) {
        cout << "MPI + SIMD 并行 Barrett Radix-4 NTT, ranks = " << ctx.size << '\n';
#ifdef __ARM_NEON
        cout << "SIMD优化: ARM NEON 已启用\n";
#else
        cout << "SIMD优化: 未启用（回退到标量实现）\n";
#endif
        cout << "支持混合Radix-2/Radix-4算法\n";
        cout << "对比标量和SIMD优化的NTT性能\n";
        cout << "=" << string(60, '=') << '\n';
        cout.flush();
    }

    for(int id = first; id <= last; ++id) {
        int n, p;

        if(ctx.rank == 0) {
            fRead(a, b, &n, &p, id);
        }

        MPI_Bcast(&n, 1, MPI_INT, 0, ctx.comm);
        MPI_Bcast(&p, 1, MPI_INT, 0, ctx.comm);

        MPI_Bcast(a, n, MPI_INT, 0, ctx.comm);
        MPI_Bcast(b, n, MPI_INT, 0, ctx.comm);

        MPI_Barrier(ctx.comm);

        auto t0_simd = chrono::high_resolution_clock::now();
        poly_multiply_radix4_simd_parallel(a, b, ab, n, p, ctx);
        MPI_Barrier(ctx.comm);
        auto t1_simd = chrono::high_resolution_clock::now();
        double us_simd = chrono::duration<double, std::micro>(t1_simd - t0_simd).count();

        if(ctx.rank == 0) {
            fCheck(ab, n, id);
        }

        MPI_Barrier(ctx.comm);
        auto t0_scalar = chrono::high_resolution_clock::now();
        poly_multiply_radix4_scalar_parallel(a, b, ab, n, p, ctx);
        MPI_Barrier(ctx.comm);
        auto t1_scalar = chrono::high_resolution_clock::now();
        double us_scalar = chrono::duration<double, std::micro>(t1_scalar - t0_scalar).count();

        if(ctx.rank == 0) {
            bool odd = __builtin_ctz(n) & 1;
            cout << "测试用例 " << id << " (n=" << n << ", p=" << p << "):\n";
            cout << "  算法类型:  " << (odd ? "混合Radix-2+Radix-4" : "纯Radix-4") << "\n";
            cout << "  标量NTT:   " << fixed << setprecision(2) << us_scalar << " us\n";
            cout << "  SIMD NTT:  " << fixed << setprecision(2) << us_simd << " us\n";
            cout << "  加速比:    " << fixed << setprecision(2) << us_scalar/us_simd << "x\n";
            cout << "  效率:      " << fixed << setprecision(2)
                 << (us_scalar/us_simd)/ctx.size*100 << "%\n";
            cout << string(60, '-') << '\n';
            cout.flush();

            fWrite(ab, n, id);
        }
    }

    MPI_Barrier(ctx.comm);
    if(ctx.rank == 0) {
        cout << "全部用例处理完成。\n";
        cout << "并行化策略: MPI数据级并行 + SIMD向量化优化\n";
        cout << "优化技术: Barrett快速取模 + NEON向量化Radix-4蝶形运算\n";
        cout << "算法特性: 自适应Radix-2/Radix-4混合算法\n";
    }

    MPI_Finalize();
    return 0;
}
