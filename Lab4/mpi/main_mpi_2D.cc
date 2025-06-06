#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;

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
        rev[i] = (rev[i>>1] >> 1) | ((i & 1) ? (lim >> 1) : 0);
}

void ntt2DBlock(int *a, int lim, int opt, int p,
                MPI_Comm cartComm,
                int Prow, int Pcol,
                int prow, int pcol) {
    int worldRank; MPI_Comm_rank(cartComm, &worldRank);
    static vector<int> rev; rev.resize(lim);
    if (worldRank == 0) {
        get_rev(rev.data(), lim);
        for (int i = 0; i < lim; ++i)
            if (i < rev[i]) swap(a[i], a[rev[i]]);
    }
    MPI_Bcast(a, lim, MPI_INT, 0, cartComm);
    vector<int> local(lim);
    for (int len = 2; len <= lim; len <<= 1) {
        int m = len >> 1;
        int blocks = lim / len;
        int rows_base = blocks / Prow, rows_rem = blocks % Prow;
        int rowCnt = rows_base + (prow < rows_rem);
        int rowBeg = prow * rows_base + min(prow, rows_rem);
        int cols_base = m / Pcol, cols_rem = m % Pcol;
        int colCnt = cols_base + (pcol < cols_rem);
        int colBeg = pcol * cols_base + min(pcol, cols_rem);
        fill(local.begin(), local.end(), 0);
        int wn = qpow(3, (p - 1) / len, p);
        if (opt == -1) wn = qpow(wn, p - 2, p);
        for (int r = 0; r < rowCnt; ++r) {
            int globalRow = rowBeg + r;
            int off = globalRow * len;
            for (int c = 0; c < colCnt; ++c) {
                int j = colBeg + c;
                int wj = qpow(wn, j, p);
                int u = a[off + j];
                int v = 1LL * a[off + j + m] * wj % p;
                int A0 = (u + v) % p;
                int A1 = (u - v + p) % p;
                local[off + j] = A0;
                local[off + j + m] = A1;
            }
        }
        MPI_Reduce(local.data(), a, lim, MPI_INT, MPI_SUM, 0, cartComm);
        if (worldRank == 0)
            for (int i = 0; i < lim; ++i) a[i] %= p;
        MPI_Bcast(a, lim, MPI_INT, 0, cartComm);
    }
    if (opt == -1) {
        int worldRank; MPI_Comm_rank(cartComm, &worldRank);
        if (worldRank == 0) {
            int inv = qpow(lim, p - 2, p);
            for (int i = 0; i < lim; ++i)
                a[i] = 1LL * a[i] * inv % p;
        }
        MPI_Bcast(a, lim, MPI_INT, 0, cartComm);
    }
}

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
        if (x != ab[i]) { return; }
    }
}

void polyMul2D(int *a, int *b, int *ab, int n, int p,
               MPI_Comm cartComm, int Prow, int Pcol,
               int prow, int pcol) {
    int lim = 1; while (lim < 2 * n) lim <<= 1;
    vector<int> A(lim), B(lim);
    int worldRank; MPI_Comm_rank(cartComm, &worldRank);
    if (worldRank == 0) {
        fill(A.begin(), A.end(), 0);
        fill(B.begin(), B.end(), 0);
        copy(a, a + n, A.begin());
        copy(b, b + n, B.begin());
    }
    MPI_Bcast(A.data(), lim, MPI_INT, 0, cartComm);
    MPI_Bcast(B.data(), lim, MPI_INT, 0, cartComm);
    ntt2DBlock(A.data(), lim, +1, p, cartComm, Prow, Pcol, prow, pcol);
    ntt2DBlock(B.data(), lim, +1, p, cartComm, Prow, Pcol, prow, pcol);
    for (int i = 0; i < lim; ++i)
        A[i] = 1LL * A[i] * B[i] % p;
    ntt2DBlock(A.data(), lim, -1, p, cartComm, Prow, Pcol, prow, pcol);
    if (worldRank == 0)
        memcpy(ab, A.data(), sizeof(int) * (2 * n - 1));
}

static int a[300000], b[300000], ab[600000];

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    int dims[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dims);
    int Prow = dims[0], Pcol = dims[1];
    int periods[2] = {0, 0};
    MPI_Comm cartComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cartComm);
    int coords[2];
    MPI_Cart_coords(cartComm, worldRank, 2, coords);
    int prow = coords[0], pcol = coords[1];
    for (int id = 0; id <= 3; ++id) {
        int n, p;
        if (worldRank == 0) fRead(a, b, &n, &p, id);
        MPI_Bcast(&n, 1, MPI_INT, 0, cartComm);
        MPI_Bcast(&p, 1, MPI_INT, 0, cartComm);
        MPI_Barrier(cartComm);
        double t0 = MPI_Wtime();
        polyMul2D(a, b, ab, n, p, cartComm, Prow, Pcol, prow, pcol);
        MPI_Barrier(cartComm);
        double t1 = MPI_Wtime();
        if (worldRank == 0) {
            fCheck(ab, n, id);
            double us = (t1 - t0) * 1e6;
            std::cout << "average latency for n=" << n
                      << " p=" << p << " : " << us << " us\n";
        }
    }
    MPI_Finalize();
    return 0;
}
