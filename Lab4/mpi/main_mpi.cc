#include <mpi.h>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <sys/time.h>

// 快速幂
int qpow(int x, int y, int p) {
  long long res = 1, base = x % p;
  while (y) {
    if (y & 1) res = res * base % p;
    base = base * base % p;
    y >>= 1;
  }
  return (int)res;
}

// 生成位反转表
void get_rev(int *rev, int lim) {
  for (int i = 0; i < lim; ++i) {
    rev[i] = (rev[i>>1] >> 1) | ((i&1) ? (lim>>1) : 0);
  }
}

// MPI 并行 NTT
void ntt_mpi(int *a, int lim, int opt, int p, int rank, int size) {
  // --- 位反转（只 rank0 做 swap，然后广播） ---
  std::vector<int> rev(lim);
  get_rev(rev.data(), lim);
  if (rank == 0) {
    for (int i = 0; i < lim; ++i)
      if (i < rev[i]) std::swap(a[i], a[rev[i]]);
  }
  MPI_Bcast(a, lim, MPI_INT, 0, MPI_COMM_WORLD);

for (int len = 2; len <= lim; len <<= 1) {
    int m  = len >> 1;
    int wn = qpow(3, (p - 1) / len, p);
    if (opt == -1) wn = qpow(wn, p - 2, p);

    int blocks = lim / len;
    int base   = blocks / size;
    int rem    = blocks % size;
    int my_blocks = base + (rank < rem ? 1 : 0);
    int start_blk = rank * base + std::min(rank, rem);

    // 本地做蝶形
    for (int b = 0; b < my_blocks; ++b) {
      int offset = (start_blk + b) * len;
      int w = 1;
      for (int j = 0; j < m; ++j) {
        int u = a[offset + j];
        int v = (int)(1LL * a[offset + j + m] * w % p);
        a[offset + j]     = (u + v) % p;
        a[offset + j + m] = (u - v + p) % p;
        w = (int)(1LL * w * wn % p);
      }
    }

    // 构造 recvcounts/displs
    std::vector<int> recvcounts(size), displs(size);
    for (int r = 0; r < size; ++r) {
      int cnt = base + (r < rem ? 1 : 0);
      recvcounts[r] = cnt * len;
      displs[r]      = (r * base + std::min(r, rem)) * len;
    }

    // 用一个临时缓冲区存放自己要发送的部分
    int sendcount = my_blocks * len;
    std::vector<int> tmp(sendcount);
    if (sendcount > 0) {
      memcpy(tmp.data(),
             a + start_blk * len,
             sendcount * sizeof(int));
    }

    // 交换到 a[] 中
    MPI_Allgatherv(
      tmp.data(), sendcount, MPI_INT,
      a,           recvcounts.data(), displs.data(), MPI_INT,
      MPI_COMM_WORLD
    );
  }
  // --- 逆变换最后一步：除以 lim，只在 rank0 做，并广播 ---
  if (opt == -1) {
    if (rank == 0) {
      int inv = qpow(lim, p - 2, p);
      for (int i = 0; i < lim; ++i)
        a[i] = (int)(1LL * a[i] * inv % p);
    }
    MPI_Bcast(a, lim, MPI_INT, 0, MPI_COMM_WORLD);
  }
}

// 并行多项式乘法
void poly_multiply_mpi(int *a, int *b, int *ab,
                       int n, int p,
                       int rank, int size) {
  int lim = 1;
  while (lim < 2 * n) lim <<= 1;

  std::vector<int> A(lim), B(lim);
  if (rank == 0) {
    memset(A.data(), 0, lim * sizeof(int));
    memset(B.data(), 0, lim * sizeof(int));
    for (int i = 0; i < n; ++i) {
      A[i] = a[i];
      B[i] = b[i];
    }
  }
  // 广播初始数组
  MPI_Bcast(A.data(), lim, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(B.data(), lim, MPI_INT, 0, MPI_COMM_WORLD);

  // 正变换
  ntt_mpi(A.data(), lim, +1, p, rank, size);
  ntt_mpi(B.data(), lim, +1, p, rank, size);

  // 点乘：每个 rank 本地对全数组做乘法
  for (int i = 0; i < lim; ++i) {
    A[i] = (int)(1LL * A[i] * B[i] % p);
  }

  // 逆变换
  ntt_mpi(A.data(), lim, -1, p, rank, size);

  // 只在 rank0 复制结果并校验／输出
  if (rank == 0) {
    for (int i = 0; i < 2 * n - 1; ++i)
      ab[i] = A[i];
  }
}


// 从文件读入多项式
void fRead(int *a, int *b, int *n, int *p, int input_id) {
    std::string path = "/nttdata/" + std::to_string(input_id) + ".in";
    std::ifstream fin(path);
    fin >> *n >> *p;
    for (int i = 0; i < *n; i++) fin >> a[i];
    for (int i = 0; i < *n; i++) fin >> b[i];
}

// 校验
void fCheck(int *ab, int n, int input_id) {
    std::string path = "/nttdata/" + std::to_string(input_id) + ".out";
    std::ifstream fin(path);
    for (int i = 0; i < 2 * n - 1; i++) {
        int x; fin >> x;
        if (x != ab[i]) {
            std::cout << "多项式乘法结果错误 (id=" << input_id << ")\n";
            return;
        }
    }
    std::cout << "多项式乘法结果正确 (id=" << input_id << ")\n";
}

// 写出（可选）
void fWrite(int *ab, int n, int input_id) {
    std::string path = "files/" + std::to_string(input_id) + ".out";
    std::ofstream fout(path);
    for (int i = 0; i < 2 * n - 1; i++) fout << ab[i] << "\n";
}

int a[300000], b[300000], ab[600000];

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int test_begin = 0, test_end = 3;
    for (int id = test_begin; id <= test_end; ++id) {
        int n, p;
        if (rank == 0) {
            fRead(a, b, &n, &p, id);
        }
        // 广播 n,p
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        // 真正的并行多项式乘法
        poly_multiply_mpi(a, b, ab, n, p, rank, size);

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (rank == 0) {
            fCheck(ab, n, id);
            double us = (t1 - t0) * 1e6;
            std::cout << "average latency for n=" << n
                      << " p=" << p << " : " << us << " us\n";
            fWrite(ab, n, id);
        }
    }

    MPI_Finalize();
    return 0;
}
