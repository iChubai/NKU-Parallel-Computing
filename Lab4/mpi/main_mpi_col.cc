/*
 * ===========================================
 * 文件名 main_mpi_column.cc
 * 描述   MPI 并行 NTT —— 列块划分 (Column‑Partition)
 * 编译   mpicxx -O3 -march=native -std=c++17 main_mpi_column.cc -o ntt_mpi
 * 运行   mpirun -np 4 ./ntt_mpi
 * ===========================================
 */
#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;

/* ---------- 工具函数 ---------- */
static int qpow(int x, long long y, int p) {
    long long res = 1, base = x % p;
    while (y) {
        if (y & 1) res = res * base % p;
        base = base * base % p;
        y >>= 1;
    }
    return int(res);
}
static void get_rev(int *rev, int lim) {
    for (int i = 0; i < lim; ++i)
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) ? (lim >> 1) : 0);
}

/* ---------- 列块划分 NTT ---------- */
void nttColumnPartition(int *a, int lim, int opt, int p,
                        int rank, int size) {
    /* 1) 位反转置换（rank0 执行） */
    static vector<int> rev;
    rev.resize(lim);
    if (rank == 0) {
        get_rev(rev.data(), lim);
        for (int i = 0; i < lim; ++i)
            if (i < rev[i]) swap(a[i], a[rev[i]]);
    }
    MPI_Bcast(a, lim, MPI_INT, 0, MPI_COMM_WORLD);

    /* 2) 每轮蝶形 —— 列块划分 */
    for (int len = 2; len <= lim; len <<= 1) {
        int m  = len >> 1;
        int wn = qpow(3, (p - 1) / len, p);
        if (opt == -1) wn = qpow(wn, p - 2, p);

        int cols_base = m / size, cols_rem = m % size;
        int my_cols   = cols_base + (rank < cols_rem);
        int start_col = rank * cols_base + min(rank, cols_rem);
        int blocks    = lim / len;

        /* 本地更新自己负责的列 */
        for (int c = 0; c < my_cols; ++c) {
            int j = start_col + c;
            int wj = qpow(wn, j, p);          // wn^j
            for (int b = 0; b < blocks; ++b) {
                int off = b * len;
                int u = a[off + j];
                int v = 1LL * a[off + j + m] * wj % p;
                a[off + j]     = (u + v) % p;
                a[off + j + m] = (u - v + p) % p;
            }
        }

        /* 打包发送缓冲区 */
        int sendcnt = my_cols * blocks * 2;
        vector<int> sendbuf(sendcnt);
        int idx = 0;
        for (int c = 0; c < my_cols; ++c) {
            int j = start_col + c;
            for (int b = 0; b < blocks; ++b) {
                int off = b * len;
                sendbuf[idx++] = a[off + j];
                sendbuf[idx++] = a[off + j + m];
            }
        }

        /* recvcounts / displs */
        vector<int> recvcounts(size), displs(size);
        int acc = 0;
        for (int r = 0; r < size; ++r) {
            int cnum = cols_base + (r < cols_rem);
            recvcounts[r] = cnum * blocks * 2;
            displs[r]     = acc;
            acc += recvcounts[r];
        }

        vector<int> recvbuf(acc);
        MPI_Allgatherv(sendbuf.data(), sendcnt, MPI_INT,
                       recvbuf.data(), recvcounts.data(),
                       displs.data(), MPI_INT,
                       MPI_COMM_WORLD);

        /* 展开回 a[] */
        for (int r = 0, pos = 0; r < size; ++r) {
            int cnum   = cols_base + (r < cols_rem);
            int st_col = r * cols_base + min(r, cols_rem);
            for (int c = 0; c < cnum; ++c) {
                int j = st_col + c;
                for (int b = 0; b < blocks; ++b, pos += 2) {
                    int off = b * len;
                    a[off + j]     = recvbuf[pos];
                    a[off + j + m] = recvbuf[pos + 1];
                }
            }
        }
    }

    /* 3) 逆变换最后一步除以 lim */
    if (opt == -1) {
        if (rank == 0) {
            int inv = qpow(lim, p - 2, p);
            for (int i = 0; i < lim; ++i)
                a[i] = 1LL * a[i] * inv % p;
        }
        MPI_Bcast(a, lim, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

/* ---------- 文件读写 / 校验 ---------- */
void fRead(int *a, int *b, int *n, int *p, int id) {
    string path = "/nttdata/" + to_string(id) + ".in";
    ifstream fin(path);
    fin >> *n >> *p;
    for (int i = 0; i < *n; ++i) fin >> a[i];
    for (int i = 0; i < *n; ++i) fin >> b[i];
}
void fCheck(const int *ab, int n, int id) {
    string path = "/nttdata/" + to_string(id) + ".out";
    ifstream fin(path);
    for (int i = 0; i < 2 * n - 1; ++i) {
        int x; fin >> x;
        if (x != ab[i]) {
            cout << "多项式乘法结果错误 (id=" << id << ")\n";
            return;
        }
    }
    cout << "多项式乘法结果正确 (id=" << id << ")\n";
}

/* ---------- 并行多项式乘法 ---------- */
void polyMulMPI(int *a, int *b, int *ab,
                int n, int p, int rank, int size) {
    int lim = 1; while (lim < 2 * n) lim <<= 1;

    vector<int> A(lim), B(lim);
    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            A[i] = a[i];
            B[i] = b[i];
        }
        fill(A.begin() + n, A.end(), 0);
        fill(B.begin() + n, B.end(), 0);
    }
    MPI_Bcast(A.data(), lim, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), lim, MPI_INT, 0, MPI_COMM_WORLD);

    nttColumnPartition(A.data(), lim, +1, p, rank, size);
    nttColumnPartition(B.data(), lim, +1, p, rank, size);

    for (int i = 0; i < lim; ++i)
        A[i] = 1LL * A[i] * B[i] % p;

    nttColumnPartition(A.data(), lim, -1, p, rank, size);

    if (rank == 0)
        memcpy(ab, A.data(), sizeof(int) * (2 * n - 1));
}

/* ---------- main ---------- */
static int a[300000], b[300000], ab[600000];

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int id = 0; id <= 3; ++id) {
        int n, p;
        if (rank == 0) fRead(a, b, &n, &p, id);
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        polyMulMPI(a, b, ab, n, p, rank, size);

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (rank == 0) {
            fCheck(ab, n, id);
            cout << "average latency for n=" << n
                 << " p=" << p << " : "
                 << (t1 - t0) * 1e6 << " us\n";
        }
    }

    MPI_Finalize();
    return 0;
}
