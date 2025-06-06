#include <bits/stdc++.h>
#include <mpi.h>
#include <iomanip>
#include <chrono>
using namespace std;

using u32  = uint32_t;
using u64  = uint64_t;
#if defined(_MSC_VER) && !defined(__clang__)
using u128 = unsigned __int128;
#else
using u128 = __uint128_t;
#endif

/* ---------------- 固定 I/O ---------------- */
void fRead(int *a, int *b, int *n, int *p, int input_id) {
    string path = "/nttdata/" + to_string(input_id) + ".in";
    ifstream fin(path);
    if(!fin) { cerr << "无法打开输入文件: " << path << '\n'; MPI_Abort(MPI_COMM_WORLD, 1); }
    fin >> *n >> *p;
    for (int i = 0; i < *n; ++i) fin >> a[i];
    for (int i = 0; i < *n; ++i) fin >> b[i];
}
void fCheck(int *ab, int n, int input_id) {
    string path = "/nttdata/" + to_string(input_id) + ".out";
    ifstream fin(path);
    if(!fin) { cerr << "无法打开输出文件: " << path << '\n'; MPI_Abort(MPI_COMM_WORLD, 1); }
    for (int i = 0; i < 2 * n - 1; ++i) {
        int x; fin >> x;
        if (x != ab[i]) { cout << "多项式乘法结果错误 (id="<<input_id<<")\n"; return; }
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
    inline u32 mul(u32 a,u32 b) const {
        return reduce(static_cast<u64>(a)*b);
    }
    const u32 mod;
private:
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
static void bit_reverse(vector<int>& rev,int n) {
    int lg = __builtin_ctz(n);
    rev.resize(n);
    for (int i=0;i<n;++i) rev[i]=(rev[i>>1]>>1)|((i&1)<<(lg-1));
}

struct MPIContext {
    int rank, size;
    MPI_Comm comm;
};
void distribute_data(const vector<u32>& global_data, vector<u32>& local_data,
                    int n, const MPIContext& ctx) {
    int local_size = n / ctx.size;
    int remainder = n % ctx.size;

    vector<int> sendcounts(ctx.size), displs(ctx.size);
    for(int i = 0; i < ctx.size; ++i) {
        sendcounts[i] = local_size + (i < remainder ? 1 : 0);
        displs[i] = i * local_size + min(i, remainder);
    }

    int my_count = sendcounts[ctx.rank];
    local_data.resize(my_count);

    MPI_Scatterv(global_data.data(), sendcounts.data(), displs.data(), MPI_UINT32_T,
                 local_data.data(), my_count, MPI_UINT32_T, 0, ctx.comm);
}

void gather_data(vector<u32>& global_data, const vector<u32>& local_data,
                int n, const MPIContext& ctx) {
    int local_size = n / ctx.size;
    int remainder = n % ctx.size;

    vector<int> recvcounts(ctx.size), displs(ctx.size);
    for(int i = 0; i < ctx.size; ++i) {
        recvcounts[i] = local_size + (i < remainder ? 1 : 0);
        displs[i] = i * local_size + min(i, remainder);
    }

    if(ctx.rank == 0) global_data.resize(n);

    MPI_Gatherv(local_data.data(), local_data.size(), MPI_UINT32_T,
                global_data.data(), recvcounts.data(), displs.data(), MPI_UINT32_T,
                0, ctx.comm);
}

void ntt_parallel(vector<u32>& a, bool inverse, const Barrett& br,
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
            memcpy(temp_data.data(), a.data() + start_block * len,
                   my_blocks * len * sizeof(u32));
        }

        MPI_Allgatherv(temp_data.data(), my_blocks * len, MPI_UINT32_T,
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

void ntt_serial(vector<u32>& a,bool inverse,const Barrett& br,u32 g=3) {
    int n=a.size();
    vector<int> rev; bit_reverse(rev,n);
    for(int i=0;i<n;++i) if(i<rev[i]) swap(a[i],a[rev[i]]);

    for(int len=2;len<=n;len<<=1){
        int m=len>>1;
        u32 wn=mod_pow(g,(br.mod-1)/len,br.mod);
        if(inverse) wn=mod_pow(wn,br.mod-2,br.mod);
        for(int i=0;i<n;i+=len){
            u32 w=1;
            for(int j=0;j<m;++j){
                u32 u=a[i+j];
                u32 v=br.mul(a[i+j+m],w);
                a[i+j]       = u+v>=br.mod?u+v-br.mod:u+v;
                a[i+j+m]     = u>=v?u-v:u+br.mod-v;
                w=br.mul(w,wn);
            }
        }
    }
    if(inverse){
        u32 inv_n=mod_pow(n,br.mod-2,br.mod);
        for(u32 &x:a) x=br.mul(x,inv_n);
    }
}

void poly_multiply_parallel(const int* a, const int* b, int* ab, int n, int p,
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

    MPI_Bcast(A.data(), lim, MPI_UINT32_T, 0, ctx.comm);
    MPI_Bcast(B.data(), lim, MPI_UINT32_T, 0, ctx.comm);

    ntt_parallel(A, false, br, ctx);
    ntt_parallel(B, false, br, ctx);

    for(int i = 0; i < lim; ++i) {
        A[i] = br.mul(A[i], B[i]);
    }

    ntt_parallel(A, true, br, ctx);

    if(ctx.rank == 0) {
        for(int i = 0; i < 2*n-1; ++i) {
            ab[i] = static_cast<int>(A[i]);
        }
    }
}

void poly_multiply_serial(const int* a,const int* b,int* ab,int n,int p){
    Barrett br(p);
    int lim=1; while(lim<2*n) lim<<=1;
    vector<u32> A(lim,0),B(lim,0);
    for(int i=0;i<n;++i){ A[i]=((a[i]%p)+p)%p; B[i]=((b[i]%p)+p)%p; }

    ntt_serial(A,false,br);
    ntt_serial(B,false,br);
    for(int i=0;i<lim;++i) A[i]=br.mul(A[i],B[i]);
    ntt_serial(A,true,br);

    for(int i=0;i<2*n-1;++i) ab[i]=static_cast<int>(A[i]);
}

static int a[300000],b[300000],ab[600000];

int main(int argc,char* argv[]){
    MPI_Init(&argc,&argv);

    MPIContext ctx;
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);
    ctx.comm = MPI_COMM_WORLD;

    const int first = 0;
    const int last  = 3;

    if(ctx.rank == 0){
        cout.flush();
    }

    for(int id = first; id <= last; ++id){
        int n, p;

        if(ctx.rank == 0) {
            fRead(a, b, &n, &p, id);
        }

        MPI_Bcast(&n, 1, MPI_INT, 0, ctx.comm);
        MPI_Bcast(&p, 1, MPI_INT, 0, ctx.comm);

        MPI_Bcast(a, n, MPI_INT, 0, ctx.comm);
        MPI_Bcast(b, n, MPI_INT, 0, ctx.comm);

        MPI_Barrier(ctx.comm);

        auto t0_parallel = chrono::high_resolution_clock::now();
        poly_multiply_parallel(a, b, ab, n, p, ctx);
        MPI_Barrier(ctx.comm);
        auto t1_parallel = chrono::high_resolution_clock::now();
        double us_parallel = chrono::duration<double,std::micro>(t1_parallel-t0_parallel).count();

        if(ctx.rank == 0) {
            fCheck(ab, n, id);

            auto t0_serial = chrono::high_resolution_clock::now();
            poly_multiply_serial(a, b, ab, n, p);
            auto t1_serial = chrono::high_resolution_clock::now();
            double us_serial = chrono::duration<double,std::micro>(t1_serial-t0_serial).count();

            cout << "测试用例 " << id << " (n=" << n << ", p=" << p << "):\n";
            cout << "  串行NTT:   " << fixed << setprecision(2) << us_serial << " us\n";
            cout << "  并行NTT:   " << fixed << setprecision(2) << us_parallel << " us\n";
            cout << "  加速比:    " << fixed << setprecision(2) << us_serial/us_parallel << "x\n";
            cout << "  效率:      " << fixed << setprecision(2)
                 << (us_serial/us_parallel)/ctx.size*100 << "%\n";
            cout << string(50, '-') << '\n';
            cout.flush();

            fWrite(ab, n, id);
        }
    }

    MPI_Barrier(ctx.comm);

    MPI_Finalize();
    return 0;
}
