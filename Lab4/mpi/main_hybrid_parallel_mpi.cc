#include <bits/stdc++.h>
#include <mpi.h>
#include <omp.h>

using namespace std;

void fRead(int *a, int *b, int *n, int *p, int input_id) {
    string path = "/nttdata/" + to_string(input_id) + ".in";
    ifstream fin(path);
    fin >> *n >> *p;
    for (int i = 0; i < *n; ++i) fin >> a[i];
    for (int i = 0; i < *n; ++i) fin >> b[i];
}

void fCheck(int *ab, int n, int input_id) {
    string path = "/nttdata/" + to_string(input_id) + ".out";
    ifstream fin(path);
    for (int i = 0; i < 2 * n - 1; ++i) {
        int x; fin >> x;
        if (x != ab[i]) {
            cout << "多项式乘法结果错误\n";
            return;
        }
    }
    cout << "多项式乘法结果正确\n";
}

int mod_pow(int base, int exp, int mod) {
    int result = 1;
    while (exp > 0) {
        if (exp % 2 == 1) result = (1LL * result * base) % mod;
        base = (1LL * base * base) % mod;
        exp /= 2;
    }
    return result;
}

void bitrev(int *a, int n) {
    int lg = __builtin_ctz(n);
    for (int i = 0; i < n; ++i) {
        int rev = 0;
        for (int j = 0; j < lg; ++j) {
            if (i >> j & 1) rev |= 1 << (lg - 1 - j);
        }
        if (i < rev) swap(a[i], a[rev]);
    }
}

enum ParallelStrategy {
    DATA_PARALLEL,
    TASK_PARALLEL,
    HYBRID_PARALLEL
};

struct MPIContext {
    int rank, size;
    MPI_Comm comm;
    MPIContext() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        comm = MPI_COMM_WORLD;
    }
};

void ntt_data_parallel(int *a, int n, bool inv, int p, const MPIContext& ctx) {
    if (ctx.rank == 0) bitrev(a, n);
    MPI_Bcast(a, n, MPI_INT, 0, ctx.comm);

    for (int len = 2; len <= n; len <<= 1) {
        int wn = mod_pow(3, (p - 1) / len, p);
        if (inv) wn = mod_pow(wn, p - 2, p);

        int total_blocks = n / len;
        int blocks_per_proc = total_blocks / ctx.size;
        int remainder = total_blocks % ctx.size;
        int my_blocks = blocks_per_proc + (ctx.rank < remainder ? 1 : 0);
        int start_block = ctx.rank * blocks_per_proc + min(ctx.rank, remainder);

        #pragma omp parallel for
        for (int block = 0; block < my_blocks; ++block) {
            int i = (start_block + block) * len;
            if (i >= n) continue;

            int w = 1;
            for (int j = 0; j < len / 2; ++j) {
                int u = a[i + j];
                int v = (1LL * a[i + j + len / 2] * w) % p;
                a[i + j] = (u + v) % p;
                a[i + j + len / 2] = (u - v + p) % p;
                w = (1LL * w * wn) % p;
            }
        }

        vector<int> recvcounts(ctx.size);
        vector<int> displs(ctx.size);
        for (int r = 0; r < ctx.size; ++r) {
            int r_blocks = blocks_per_proc + (r < remainder ? 1 : 0);
            recvcounts[r] = r_blocks * len;
            displs[r] = (r * blocks_per_proc + min(r, remainder)) * len;
        }

        int sendcnt = my_blocks * len;
        if (sendcnt > 0 && start_block * len < n) {
            vector<int> sendbuf(sendcnt);
            memcpy(sendbuf.data(), a + start_block * len, sizeof(int) * min(sendcnt, n - start_block * len));
            MPI_Allgatherv(sendbuf.data(), sendcnt, MPI_INT, a, recvcounts.data(), displs.data(), MPI_INT, ctx.comm);
        } else {
            MPI_Allgatherv(nullptr, 0, MPI_INT, a, recvcounts.data(), displs.data(), MPI_INT, ctx.comm);
        }
    }

    if (inv && ctx.rank == 0) {
        int invN = mod_pow(n, p - 2, p);
        for (int i = 0; i < n; ++i) a[i] = (1LL * a[i] * invN) % p;
    }
    if (inv) MPI_Bcast(a, n, MPI_INT, 0, ctx.comm);
}

void ntt_task_parallel(int *a, int n, bool inv, int p, const MPIContext& ctx) {
    if (ctx.rank == 0) {
        bitrev(a, n);

        for (int len = 2; len <= n; len <<= 1) {
            int wn = mod_pow(3, (p - 1) / len, p);
            if (inv) wn = mod_pow(wn, p - 2, p);

            #pragma omp parallel for
            for (int i = 0; i < n; i += len) {
                int w = 1;
                for (int j = 0; j < len / 2; ++j) {
                    int u = a[i + j];
                    int v = (1LL * a[i + j + len / 2] * w) % p;
                    a[i + j] = (u + v) % p;
                    a[i + j + len / 2] = (u - v + p) % p;
                    w = (1LL * w * wn) % p;
                }
            }
        }

        if (inv) {
            int invN = mod_pow(n, p - 2, p);
            #pragma omp parallel for
            for (int i = 0; i < n; ++i) a[i] = (1LL * a[i] * invN) % p;
        }
    }
    MPI_Bcast(a, n, MPI_INT, 0, ctx.comm);
}

void ntt_hybrid_parallel(int *a, int n, bool inv, int p, const MPIContext& ctx) {
    if (n <= 1024) {
        ntt_task_parallel(a, n, inv, p, ctx);
    } else {
        ntt_data_parallel(a, n, inv, p, ctx);
    }
}

void adaptive_ntt_mpi(int *a, int n, bool inv, int p, const MPIContext& ctx, ParallelStrategy strategy) {
    switch (strategy) {
        case DATA_PARALLEL:
            ntt_data_parallel(a, n, inv, p, ctx);
            break;
        case TASK_PARALLEL:
            ntt_task_parallel(a, n, inv, p, ctx);
            break;
        case HYBRID_PARALLEL:
            ntt_hybrid_parallel(a, n, inv, p, ctx);
            break;
    }
}

void poly_mul_adaptive_mpi(const int *A_const, const int *B_const, int *C, int n_orig, int p,
                          const MPIContext& ctx, ParallelStrategy strategy) {
    if (n_orig <= 0) return;

    int N = 1;
    while (N < 2 * n_orig) N <<= 1;

    vector<int> pa(N), pb(N);
    for (int i = 0; i < n_orig; ++i) {
        pa[i] = ((A_const[i] % p) + p) % p;
        pb[i] = ((B_const[i] % p) + p) % p;
    }
    for (int i = n_orig; i < N; ++i) {
        pa[i] = 0;
        pb[i] = 0;
    }

    adaptive_ntt_mpi(pa.data(), N, false, p, ctx, strategy);
    adaptive_ntt_mpi(pb.data(), N, false, p, ctx, strategy);

    if (strategy == TASK_PARALLEL && ctx.rank == 0) {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            pa[i] = (1LL * pa[i] * pb[i]) % p;
        }
    } else {
        int items_per_rank = N / ctx.size;
        int remainder = N % ctx.size;
        int my_items = items_per_rank + (ctx.rank < remainder ? 1 : 0);
        int my_start = ctx.rank * items_per_rank + min(ctx.rank, remainder);

        #pragma omp parallel for
        for (int i = 0; i < my_items; ++i) {
            int global_idx = my_start + i;
            if (global_idx < N) {
                pa[global_idx] = (1LL * pa[global_idx] * pb[global_idx]) % p;
            }
        }

        vector<int> recvcounts(ctx.size);
        vector<int> displs(ctx.size);
        int current_displ = 0;
        for (int r = 0; r < ctx.size; ++r) {
            recvcounts[r] = items_per_rank + (r < remainder ? 1 : 0);
            displs[r] = current_displ;
            current_displ += recvcounts[r];
        }

        vector<int> local_data(my_items);
        for (int i = 0; i < my_items; ++i) {
            int global_idx = my_start + i;
            local_data[i] = (global_idx < N) ? pa[global_idx] : 0;
        }

        MPI_Allgatherv(local_data.data(), my_items, MPI_INT, pa.data(),
                       recvcounts.data(), displs.data(), MPI_INT, ctx.comm);
    }

    adaptive_ntt_mpi(pa.data(), N, true, p, ctx, strategy);

    if (ctx.rank == 0) {
        int result_size = 2 * n_orig - 1;
        if (result_size > 0 && result_size <= N) {
            memcpy(C, pa.data(), sizeof(int) * result_size);
        }
    }
}

const char* strategy_name(ParallelStrategy s) {
    switch (s) {
        case DATA_PARALLEL: return "数据并行";
        case TASK_PARALLEL: return "任务并行";
        case HYBRID_PARALLEL: return "混合并行";
        default: return "未知";
    }
}

void performance_comparison(const vector<int>& a, const vector<int>& b, int n, int p, const MPIContext& ctx) {
    vector<ParallelStrategy> strategies = {DATA_PARALLEL, TASK_PARALLEL, HYBRID_PARALLEL};
    vector<int> result(2 * n - 1);

    if (ctx.rank == 0) {
        cout << "\n性能对比 (n=" << n << ", p=" << p << "):\n";
        cout << string(50, '-') << '\n';
    }

    for (auto strategy : strategies) {
        MPI_Barrier(ctx.comm);
        double t0 = MPI_Wtime();

        poly_mul_adaptive_mpi(a.data(), b.data(), result.data(), n, p, ctx, strategy);

        MPI_Barrier(ctx.comm);
        double t1 = MPI_Wtime();

        if (ctx.rank == 0) {
            cout << strategy_name(strategy) << ": "
                 << fixed << setprecision(3) << (t1 - t0) * 1000 << " ms\n";
        }
    }

    if (ctx.rank == 0) {
        cout << string(50, '-') << '\n';
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPIContext ctx;

    if (ctx.rank == 0) {
        cout << "混合并行策略 NTT MPI 实现\n";
        cout << "进程数: " << ctx.size << "\n";
#ifdef _OPENMP
        cout << "OpenMP线程数: " << omp_get_max_threads() << "\n";
#else
        cout << "OpenMP: 未启用\n";
#endif
        cout << "支持策略: 数据并行、任务并行、混合并行\n";
        cout << "========================================\n";
    }

    vector<int> a(300000, 0), b(300000, 0), ab(600000, 0);

    for (int id = 0; id <= 3; ++id) {
        int n = 0, p = 0;

        if (ctx.rank == 0) {
            fRead(a.data(), b.data(), &n, &p, id);
        }

        MPI_Bcast(&n, 1, MPI_INT, 0, ctx.comm);
        MPI_Bcast(&p, 1, MPI_INT, 0, ctx.comm);
        MPI_Bcast(a.data(), n, MPI_INT, 0, ctx.comm);
        MPI_Bcast(b.data(), n, MPI_INT, 0, ctx.comm);

        vector<int> a_vec(a.data(), a.data() + n);
        vector<int> b_vec(b.data(), b.data() + n);

        if (ctx.rank == 0) {
            cout << "\n测试用例 " << id << " (n=" << n << ", p=" << p << ")\n";
        }

        ParallelStrategy best_strategy = HYBRID_PARALLEL;
        MPI_Barrier(ctx.comm);
        double t0 = MPI_Wtime();

        poly_mul_adaptive_mpi(a_vec.data(), b_vec.data(), ab.data(), n, p, ctx, best_strategy);

        MPI_Barrier(ctx.comm);
        double t1 = MPI_Wtime();

        if (ctx.rank == 0) {
            fCheck(ab.data(), n, id);
            cout << "最优策略 (" << strategy_name(best_strategy) << "): "
                 << fixed << setprecision(3) << (t1 - t0) * 1000 << " ms\n";
        }

        performance_comparison(a_vec, b_vec, n, p, ctx);
    }

    if (ctx.rank == 0) {
        cout << "\n========================================\n";
        cout << "所有测试完成\n";
        cout << "优化特性:\n";
        cout << "  - 自适应并行策略选择\n";
        cout << "  - OpenMP线程级并行\n";
        cout << "  - MPI进程级并行\n";
        cout << "  - 数据和任务混合并行\n";
    }

    MPI_Finalize();
    return 0;
}
