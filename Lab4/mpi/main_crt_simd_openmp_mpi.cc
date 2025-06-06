#include <bits/stdc++.h>
#include <mpi.h>
#include <omp.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

using namespace std;

void fRead(int *a, int *b, int *n, int *p, int input_id) {
    string path = "/nttdata/" + to_string(input_id) + ".in";
    ifstream fin(path);
    if (!fin.is_open()) {
        cerr << "Fatal: Failed to open input file: " << path << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fin >> *n >> *p;
    for (int i = 0; i < *n; ++i) fin >> a[i];
    for (int i = 0; i < *n; ++i) fin >> b[i];
}

void fCheck(int *ab, int n, int input_id) {
    string path = "/nttdata/" + to_string(input_id) + ".out";
    ifstream fin(path);
    if (!fin.is_open()) {
        cerr << "Fatal: Failed to open output file for checking: " << path << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < 2 * n - 1; ++i) {
        int x; fin >> x;
        if (x != ab[i]) {
            cout << "多项式乘法结果错误\n";
            return;
        }
    }
    cout << "多项式乘法结果正确\n";
}

struct BarrettReducer {
    uint64_t mod;
    uint64_t im;
    BarrettReducer(uint64_t m = 1) { set_mod(m); }
    void set_mod(uint64_t m) {
        mod = m;
        im = (~0ULL) / m;
    }
    inline uint32_t reduce(uint64_t a) const {
        uint64_t q = (__uint128_t(im) * a) >> 64;
        uint64_t r = a - q * mod;
        if (r >= mod) r -= mod;
        return static_cast<uint32_t>(r);
    }
    inline uint32_t add(uint32_t a, uint32_t b) const {
        uint32_t s = a + b;
        return s >= mod ? s - mod : s;
    }
    inline uint32_t sub(uint32_t a, uint32_t b) const {
        return a >= b ? a - b : a + mod - b;
    }
    inline uint32_t mul(uint32_t a, uint32_t b) const {
        return reduce(uint64_t(a) * b);
    }
    uint32_t pow(uint32_t x, uint64_t e) const {
        uint32_t res = 1;
        while (e) {
            if (e & 1) res = mul(res, x);
            x = mul(x, x);
            e >>= 1;
        }
        return res;
    }
};

void serialNTT(vector<uint32_t>& a, bool inverse, const BarrettReducer& br) {
    const int n = a.size();
    if (n <= 1) return;

    static vector<int> rev; rev.resize(n);
    int lg = __builtin_ctz(n);
    for (int i = 0; i < n; ++i) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (lg - 1));
    }

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        if (i < rev[i]) swap(a[i], a[rev[i]]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        uint32_t wn = br.pow(3, (br.mod - 1) / len);
        if (inverse) wn = br.pow(wn, br.mod - 2);
        int half = len >> 1;
        #pragma omp parallel for
        for (int i = 0; i < n; i += len) {
            uint32_t w = 1;
            for (int j = 0; j < half; ++j) {
                uint32_t u = a[i + j];
                uint32_t v = br.mul(a[i + j + half], w);
                a[i + j]         = br.add(u, v);
                a[i + j + half]  = br.sub(u, v);
                w = br.mul(w, wn);
            }
        }
    }
    if (inverse) {
        uint32_t inv_n = br.pow(n, br.mod - 2);
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            a[i] = br.mul(a[i], inv_n);
        }
    }
}

uint64_t modinv(uint64_t a, uint64_t m) {
    uint64_t b = m, u = 1, v = 0;
    while (b) {
        uint64_t t = a / b;
        a -= t * b; swap(a, b);
        u -= t * v; swap(u, v);
    }
    return (u + m) % m;
}

#ifdef __ARM_NEON
void crtMergeThreeNEON(const vector<uint32_t>& r0,
                       const vector<uint32_t>& r1,
                       const vector<uint32_t>& r2,
                       uint64_t p0, uint64_t p1, uint64_t p2,
                       vector<uint64_t>& out) {
    const size_t m = r0.size();
    out.resize(m);

    uint64_t inv_p0_mod_p1 = modinv(p0, p1);
    uint64_t inv_p0p1_mod_p2 = modinv(p0 * p1 % p2, p2);
    __uint128_t M01 = (__uint128_t)p0 * p1;

    #pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        uint64_t t0 = r0[i];
        uint64_t t1 = ((r1[i] + p1 - t0 % p1) * inv_p0_mod_p1) % p1;
        uint64_t t0_t1 = (t0 + (__uint128_t)p0 * t1) % p2;
        uint64_t t2 = ((r2[i] + p2 - t0_t1) * inv_p0p1_mod_p2) % p2;

        __uint128_t res = t0 + (__uint128_t)p0 * t1 + M01 * t2;
        out[i] = (uint64_t)res;
    }
}
#endif

void crtMergeThreeOpenMP(const vector<uint32_t>& r0,
                         const vector<uint32_t>& r1,
                         const vector<uint32_t>& r2,
                         uint64_t p0, uint64_t p1, uint64_t p2,
                         vector<uint64_t>& out) {
    const size_t m = r0.size();
    out.resize(m);

    uint64_t inv_p0_mod_p1 = modinv(p0, p1);
    uint64_t inv_p0p1_mod_p2 = modinv(p0 * p1 % p2, p2);
    __uint128_t M01 = (__uint128_t)p0 * p1;

    #pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        uint64_t t0 = r0[i];
        uint64_t t1 = ((r1[i] + p1 - t0 % p1) * inv_p0_mod_p1) % p1;
        uint64_t t0_t1 = (t0 + (__uint128_t)p0 * t1) % p2;
        uint64_t t2 = ((r2[i] + p2 - t0_t1) * inv_p0p1_mod_p2) % p2;

        __uint128_t res = t0 + (__uint128_t)p0 * t1 + M01 * t2;
        out[i] = (uint64_t)res;
    }
}

struct MPIContext {
    int rank, size;
    MPI_Comm comm;
    MPIContext() { 
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
        MPI_Comm_size(MPI_COMM_WORLD, &size); 
        comm = MPI_COMM_WORLD; 
    }
};

void hybridCRTMultiply(const vector<int>& a,
                       const vector<int>& b,
                       vector<int>& result_mod_p,
                       int n,
                       uint32_t p_target,
                       const MPIContext& ctx) {
    constexpr uint32_t mod[3] = {998244353u, 1004535809u, 469762049u};
    const int MODS = 3;

    vector<int> my_mod_idx;
    for (int i = ctx.rank; i < MODS; i += ctx.size) my_mod_idx.push_back(i);

    int conv_len = 2 * n - 1;
    vector<uint32_t> r0(conv_len, 0), r1(conv_len, 0), r2(conv_len, 0);

    for (int idx : my_mod_idx) {
        BarrettReducer br(mod[idx]);
        vector<uint32_t> A(a.begin(), a.end()), B(b.begin(), b.end());

        int lim = 1; while (lim < 2 * n) lim <<= 1;
        A.resize(lim); B.resize(lim);

        serialNTT(A, false, br);
        serialNTT(B, false, br);
        for (int i = 0; i < lim; ++i) A[i] = br.mul(A[i], B[i]);
        serialNTT(A, true, br);

        for (int i = 0; i < conv_len; ++i) {
            if (idx == 0) r0[i] = A[i];
            else if (idx == 1) r1[i] = A[i];
            else r2[i] = A[i];
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, r0.data(), conv_len, MPI_UINT32_T, MPI_SUM, ctx.comm);
    MPI_Allreduce(MPI_IN_PLACE, r1.data(), conv_len, MPI_UINT32_T, MPI_SUM, ctx.comm);
    MPI_Allreduce(MPI_IN_PLACE, r2.data(), conv_len, MPI_UINT32_T, MPI_SUM, ctx.comm);

    if (ctx.rank == 0) {
        result_mod_p.resize(conv_len);

        uint64_t p0 = mod[0], p1 = mod[1], p2 = mod[2];
        uint64_t inv_p0_mod_p1 = modinv(p0, p1);
        uint64_t inv_p0p1_mod_p2 = modinv(p0 * p1 % p2, p2);
        __uint128_t M01 = (__uint128_t)p0 * p1;

        for (int i = 0; i < conv_len; ++i) {
            uint64_t x0 = r0[i];
            uint64_t t1 = ((r1[i] + p1 - x0 % p1) * inv_p0_mod_p1) % p1;
            uint64_t x0_p0t1 = (x0 + (__uint128_t)p0 * t1) % p2;
            uint64_t t2 = ((r2[i] + p2 - x0_p0t1) * inv_p0p1_mod_p2) % p2;

            __uint128_t big = x0 + (__uint128_t)p0 * t1 + M01 * t2;
            result_mod_p[i] = int(big % p_target);
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPIContext ctx;

    if (ctx.rank == 0) {
        cout << "CRT + SIMD NEON + OpenMP + MPI 混合并行实现\n";
        cout << "进程数: " << ctx.size << "\n";
#ifdef __ARM_NEON
        cout << "NEON SIMD: 启用\n";
#else
        cout << "NEON SIMD: 未启用\n";
#endif
#ifdef _OPENMP
        cout << "OpenMP线程数: " << omp_get_max_threads() << "\n";
#else
        cout << "OpenMP: 未启用\n";
#endif
        cout << "========================================\n";
    }

    int a_arr[300000], b_arr[300000];
    int test_begin = 0, test_end = 3;

    for (int id = test_begin; id <= test_end; ++id) {
        int n = 0, p_test = 0;
        if (ctx.rank == 0) fRead(a_arr, b_arr, &n, &p_test, id);
        MPI_Bcast(&n, 1, MPI_INT, 0, ctx.comm);
        MPI_Bcast(&p_test, 1, MPI_INT, 0, ctx.comm);
        MPI_Bcast(a_arr, n, MPI_INT, 0, ctx.comm);
        MPI_Bcast(b_arr, n, MPI_INT, 0, ctx.comm);

        vector<int> a(a_arr, a_arr + n);
        vector<int> b(b_arr, b_arr + n);
        vector<int> ab_mod;

        MPI_Barrier(ctx.comm);
        double t0 = MPI_Wtime();
        hybridCRTMultiply(a, b, ab_mod, n, p_test, ctx);
        double t1 = MPI_Wtime();

        if (ctx.rank == 0) {
            fCheck(ab_mod.data(), n, id);
            cout << "测试用例 " << id << " (n=" << n << ", p=" << p_test << "): "
                 << fixed << setprecision(3) << (t1 - t0) * 1000 << " ms\n";
        }
    }

    MPI_Finalize();
    return 0;
}
