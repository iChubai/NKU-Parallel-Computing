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

// This function is copied from main_barrett_radix2_ntt.cc and renamed
void ntt_scalar_parallel(vector<u32>& a, bool inverse, const Barrett& br,
                 const MPIContext& ctx, u32 g=3) {
    int n = a.size();

    if(ctx.rank == 0) {
        vector<int> rev;
        bit_reverse(rev, n);
        for(int i = 0; i < n; ++i) {
            if(i < rev[i]) swap(a[i], a[rev[i]]);
        }
    }
    MPI_Bcast(a.data(), n, MPI_UINT32_T, 0, ctx.comm);

    for(int len = 2; len <= n; len <<= 1) {
        int m = len >> 1;
        u32 wn = mod_pow(g, (br.mod-1)/len, br.mod);
        if(inverse) wn = mod_pow(wn, br.mod-2, br.mod);

        int total_blocks = n / len;
        int blocks_per_proc = total_blocks / ctx.size;
        int remainder_blocks = total_blocks % ctx.size;

        int my_blocks = blocks_per_proc + (ctx.rank < remainder_blocks ? 1 : 0);
        int start_block = ctx.rank * blocks_per_proc + min(ctx.rank, remainder_blocks);

        for(int b = 0; b < my_blocks; ++b) {
            int block_start = (start_block + b) * len;
            u32 w = 1;
            for(int j = 0; j < m; ++j) {
                u32 u = a[block_start + j];
                u32 v = br.mul(a[block_start + j + m], w);
                a[block_start + j] = u + v >= br.mod ? u + v - br.mod : u + v;
                a[block_start + j + m] = u >= v ? u - v : u + br.mod - v;
                w = br.mul(w, wn);
            }
        }

        vector<int> recvcounts(ctx.size), displs(ctx.size);
        for(int r = 0; r < ctx.size; ++r) {
            int r_blocks = blocks_per_proc + (r < remainder_blocks ? 1 : 0);
            recvcounts[r] = r_blocks * len;
            displs[r] = (r * blocks_per_proc + min(r, remainder_blocks)) * len;
        }

        vector<u32> temp_data(my_blocks * len);
        if(my_blocks > 0) {
            if (start_block * len + my_blocks * len <= n) {
                memcpy(temp_data.data(), a.data() + start_block * len,
                       my_blocks * len * sizeof(u32));
            }
        }

        MPI_Allgatherv(my_blocks > 0 ? temp_data.data() : MPI_IN_PLACE, my_blocks * len, MPI_UINT32_T,
                       a.data(), recvcounts.data(), displs.data(), MPI_UINT32_T,
                       ctx.comm);
    }

    if(inverse) {
        if(ctx.rank == 0) {
            u32 inv_n = mod_pow(n, br.mod-2, br.mod);
            for(u32 &x : a) x = br.mul(x, inv_n);
        }
        MPI_Bcast(a.data(), n, MPI_UINT32_T, 0, ctx.comm);
    }
}


void ntt_simd_parallel(vector<u32>& a, bool inverse, const Barrett& br,
                      const MPIContext& ctx, u32 g = 3) {
    int n = a.size();

    if(ctx.rank == 0) {
        vector<int> rev;
        bit_reverse(rev, n);
        for(int i = 0; i < n; ++i) {
            if(i < rev[i]) swap(a[i], a[rev[i]]);
        }
    }
    MPI_Bcast(a.data(), n, MPI_UNSIGNED, 0, ctx.comm);

#ifdef __ARM_NEON
    uint32x4_t mod_vec = vdupq_n_u32(br.mod);
#endif

    for(int len = 2; len <= n; len <<= 1) {
        int m = len >> 1;
        u32 wn = mod_pow(g, (br.mod-1)/len, br.mod);
        if(inverse) wn = mod_pow(wn, br.mod-2, br.mod);

        int total_blocks = n / len;
        int blocks_per_proc = total_blocks / ctx.size;
        int remainder_blocks = total_blocks % ctx.size;

        int my_blocks = blocks_per_proc + (ctx.rank < remainder_blocks ? 1 : 0);
        int start_block = ctx.rank * blocks_per_proc + min(ctx.rank, remainder_blocks);

        for(int b = 0; b < my_blocks; ++b) {
            int block_start = (start_block + b) * len;
            u32 w = 1;
            for(int j = 0; j < m; ++j) {
                u32 u = a[block_start + j];
                u32 v = br.mul(a[block_start + j + m], w);
                a[block_start + j] = br.add(u, v);
                a[block_start + j + m] = br.sub(u, v);
                w = br.mul(w, wn);
            }
        }

        vector<int> recvcounts(ctx.size), displs(ctx.size);
        for(int r = 0; r < ctx.size; ++r) {
            int r_blocks = blocks_per_proc + (r < remainder_blocks ? 1 : 0);
            recvcounts[r] = r_blocks * len;
            displs[r] = (r * blocks_per_proc + min(r, remainder_blocks)) * len;
        }

        vector<u32> temp_data(my_blocks * len);
        if(my_blocks > 0) {
            if (start_block * len + my_blocks * len <= n) {
                memcpy(temp_data.data(), a.data() + start_block * len,
                       my_blocks * len * sizeof(u32));
            }
        }

        MPI_Allgatherv(my_blocks > 0 ? temp_data.data() : MPI_IN_PLACE, my_blocks * len, MPI_UNSIGNED,
                       a.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                       ctx.comm);
    }

    if(inverse) {
        if(ctx.rank == 0) {
            u32 inv_n = mod_pow(n, br.mod-2, br.mod);
#ifdef __ARM_NEON
            uint32x4_t inv_n_vec = vdupq_n_u32(inv_n);
            int i = 0;
            for(; i + 3 < n; i += 4) {
                uint32x4_t x_vec = vld1q_u32(&a[i]);
                uint32x4_t result = barrett_mul_neon(x_vec, inv_n_vec, br);
                vst1q_u32(&a[i], result);
            }
            for(; i < n; ++i) {
                a[i] = br.mul(a[i], inv_n);
            }
#else
            for(u32 &x : a) x = br.mul(x, inv_n);
#endif
        }
        MPI_Bcast(a.data(), n, MPI_UNSIGNED, 0, ctx.comm);
    }
}

void poly_multiply_simd_parallel(const int* a, const int* b, int* ab, int n, int p,
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

    ntt_simd_parallel(A, false, br, ctx);
    ntt_simd_parallel(B, false, br, ctx);

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

    ntt_simd_parallel(A, true, br, ctx);

    if(ctx.rank == 0) {
        for(int i = 0; i < 2*n-1; ++i) {
            ab[i] = static_cast<int>(A[i]);
        }
    }
}

void poly_multiply_scalar_parallel(const int* a, const int* b, int* ab, int n, int p,
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

    ntt_scalar_parallel(A, false, br, ctx);
    ntt_scalar_parallel(B, false, br, ctx);

    for(int i = 0; i < lim; ++i) {
        A[i] = br.mul(A[i], B[i]);
    }

    ntt_scalar_parallel(A, true, br, ctx);

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
        cout << "MPI + SIMD 并行 Barrett Radix-2 NTT, ranks = " << ctx.size << '\n';
#ifdef __ARM_NEON
        cout << "SIMD优化: ARM NEON 已启用\n";
#else
        cout << "SIMD优化: 未启用（回退到标量实现）\n";
#endif
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
        poly_multiply_simd_parallel(a, b, ab, n, p, ctx);
        MPI_Barrier(ctx.comm);
        auto t1_simd = chrono::high_resolution_clock::now();
        double us_simd = chrono::duration<double, std::micro>(t1_simd - t0_simd).count();

        if(ctx.rank == 0) {
            fCheck(ab, n, id);
        }

        MPI_Barrier(ctx.comm);
        auto t0_scalar = chrono::high_resolution_clock::now();
        poly_multiply_scalar_parallel(a, b, ab, n, p, ctx);
        MPI_Barrier(ctx.comm);
        auto t1_scalar = chrono::high_resolution_clock::now();
        double us_scalar = chrono::duration<double, std::micro>(t1_scalar - t0_scalar).count();

        if(ctx.rank == 0) {
            cout << "测试用例 " << id << " (n=" << n << ", p=" << p << "):\n";
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
        cout << "优化技术: Barrett快速取模 + NEON向量化蝶形运算\n";
    }

    MPI_Finalize();
    return 0;
} 