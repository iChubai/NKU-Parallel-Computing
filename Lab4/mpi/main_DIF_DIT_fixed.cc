#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <mpi.h>
#include <vector>

typedef long long ll;

/**
 * 将输出结果写入文件
 * @param ab 结果数组
 * @param n 多项式长度
 * @param input_id 测试用例ID
 */
void fWrite(int *ab, int n, int input_id) {
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    std::ofstream fout(strout);
    int output_count = (n == 0) ? 0 : (2 * n - 1);
    for (int i = 0; i < output_count; i++) {
        fout << ab[i] << '\n';
    }
    fout.close();
}

/**
 * 校验计算结果与标准答案是否一致
 * @param ab 计算结果数组
 * @param n 多项式长度
 * @param input_id 测试用例ID
 * @param rank 进程ID
 */
void fCheck(int *ab, int n, int input_id, int rank) {
    if (rank == 0) {
        if (n <= 0) {
            return;
        }
        std::string path = "/nttdata/" + std::to_string(input_id) + ".out";
        std::ifstream fin(path);
        if (!fin) {
            std::cerr << "Rank 0, ID " << input_id << ": fCheck - 无法打开参考文件 " << path << std::endl;
            return;
        }

        bool match = true;
        int result_len = 2 * n - 1;
        for (int i = 0; i < result_len; ++i) {
            int expected_val;
            if (!(fin >> expected_val)) {
                std::cerr << "Rank 0, ID " << input_id << ": fCheck - 读取参考文件时发生错误 (i=" << i << ")." << std::endl;
                match = false;
                break;
            }
            if (ab[i] != expected_val) {
                match = false;
            }
        }
        fin.close();

        if (match) {
            std::cout << "多项式乘法结果正确 (id=" << input_id << ")" << std::endl;
        } else {
            std::cout << "多项式乘法结果错误 (id=" << input_id << ")" << std::endl;
        }
    }
}

/**
 * 快速幂模运算
 * @param a 底数
 * @param b 指数
 * @param P 模数
 * @return a^b mod P
 */
inline ll fpow(ll a, ll b, int P) {
    ll res = 1;
    a %= P;
    if (a < 0) a += P;
    for (; b; b >>= 1) {
        if (b & 1) res = (res * a) % P;
        a = (a * a) % P;
    }
    return res;
}

/**
 * 计算 NTT 所需的单位根幂次
 * @param w 存储单位根幂次的数组
 * @param G 变换长度
 * @param P 模数
 * @param gen 原根
 */
void calc_powg(int w[], int G, int P, int gen) {
    w[0] = 1;
    ll f;
    const int g_root = fpow(gen, (P - 1) / G, P);
    for (int t = 0; (1 << (t + 1)) < G; ++t) {
        f = w[1 << t] = fpow(g_root, G >> (t + 2), P);
        for (int x = 1 << t; x < 1 << (t + 1); ++x) {
            w[x] = (ll)f * w[x - (1 << t)] % P;
        }
    }
}

/**
 * 将全  seg_idx 数按照块状分配到各个进程
 * @param total_segments 总段数
 * @param rank 进程 ID
 * @param size 进程总数
 * @param start_seg 返回：当前进程负责的第一个段索引
 * @param local_count 返回：当前进程负责的段数
 */
void block_partition(int total_segments, int rank, int size, int &start_seg, int &local_count) {
    int base = total_segments / size;
    int rem = total_segments % size;
    if (rank < rem) {
        local_count = base + 1;
        start_seg = rank * local_count;
    } else {
        local_count = base;
        start_seg = rem * (base + 1) + (rank - rem) * base;
    }
}

/**
 * DIF (Decimation In Frequency) NTT 变换 - MPI 并行版（块状分配 & 
 * 用一次 MPI_Allgatherv 仅交换必要段）
 * @param f 待变换数组
 * @param l 长度的对数 (2^l = 变换长度)
 * @param P 模数
 * @param w 单位根幂次数组
 * @param rank MPI 进程 ID
 * @param size MPI 进程总数
 */
void DIF(int f[], int l, int P, int w[], int rank, int size) {
    int lim = 1 << l;
    ll g, h;

    // 用于打包本地段和接收所有进程段的缓冲区
    std::vector<int> sendbuf, recvbuf;

    for (int len = lim; len > 1; len >>= 1) {
        int total_segments = lim / len;

        // 计算块分配
        int start_seg, local_count;
        block_partition(total_segments, rank, size, start_seg, local_count);
        int send_elems = local_count * len; // 本地片段总长度

        // 本地蝶形更新
        for (int seg = start_seg; seg < start_seg + local_count; ++seg) {
            int st = seg * len;
            int t = seg;
            for (int i = st; i < st + len / 2; ++i) {
                g = f[i];
                h = (ll)f[i + len / 2] * w[t] % P;
                f[i] = (g + h) % P;
                f[i + len / 2] = (P + g - h) % P;
            }
        }

        // 准备 sendbuf：将本地所有段压缩到一个连续缓冲区
        sendbuf.resize(send_elems);
        {
            int idx = 0;
            for (int seg = start_seg; seg < start_seg + local_count; ++seg) {
                int st = seg * len;
                std::memcpy(&sendbuf[idx], f + st, len * sizeof(int));
                idx += len;
            }
        }

        // 计算各进程的接收计数和偏移
        std::vector<int> recvcounts(size), displs(size);
        for (int p = 0; p < size; ++p) {
            int ps, pc;
            block_partition(total_segments, p, size, ps, pc);
            recvcounts[p] = pc * len;
        }
        displs[0] = 0;
        for (int p = 1; p < size; ++p) {
            displs[p] = displs[p - 1] + recvcounts[p - 1];
        }
        int total_recv = 0;
        for (int p = 0; p < size; ++p) total_recv += recvcounts[p];
        recvbuf.resize(total_recv);

        // 用 Allgatherv 交换所有进程打包后的片段
        MPI_Allgatherv(
            sendbuf.data(), send_elems, MPI_INT,
            recvbuf.data(), recvcounts.data(), displs.data(), MPI_INT,
            MPI_COMM_WORLD
        );

        // 将 recvbuf 解包回 f 的对应段位置
        {
            int offset = 0;
            for (int p = 0; p < size; ++p) {
                int ps, pc;
                block_partition(total_segments, p, size, ps, pc);
                // ps .. ps+pc-1 段是进程 p 发来的片段
                for (int k = 0; k < pc; ++k) {
                    int seg_idx = ps + k;
                    int st = seg_idx * len;
                    std::memcpy(f + st, &recvbuf[offset + k * len], len * sizeof(int));
                }
                offset += pc * len;
            }
        }
    }
}

/**
 * DIT (Decimation In Time) INTT 逆变换 - MPI 并行版（块状分配 & 
 * 用一次 MPI_Allgatherv 仅交换必要段）
 * @param f 待逆变换数组
 * @param l 长度的对数 (2^l = 变换长度)
 * @param P 模数
 * @param w 单位根幂次数组
 * @param rank MPI 进程 ID
 * @param size MPI 进程总数
 */
void DIT(int f[], int l, int P, int w[], int rank, int size) {
    int lim = 1 << l;
    ll g, h;

    std::vector<int> sendbuf, recvbuf;

    for (int len = 2; len <= lim; len <<= 1) {
        int total_segments = lim / len;

        // 计算块分配
        int start_seg, local_count;
        block_partition(total_segments, rank, size, start_seg, local_count);
        int send_elems = local_count * len;

        // 本地蝶形更新
        for (int seg = start_seg; seg < start_seg + local_count; ++seg) {
            int st = seg * len;
            int t = seg;
            for (int i = st; i < st + len / 2; ++i) {
                g = f[i];
                h = f[i + len / 2];
                f[i] = (g + h) % P;
                f[i + len / 2] = (ll)(P + g - h) % P * w[t] % P;
            }
        }

        // 准备 sendbuf：将本地所有段压缩到一个连续缓冲区
        sendbuf.resize(send_elems);
        {
            int idx = 0;
            for (int seg = start_seg; seg < start_seg + local_count; ++seg) {
                int st = seg * len;
                std::memcpy(&sendbuf[idx], f + st, len * sizeof(int));
                idx += len;
            }
        }

        // 计算各进程的接收计数和偏移
        std::vector<int> recvcounts(size), displs(size);
        for (int p = 0; p < size; ++p) {
            int ps, pc;
            block_partition(total_segments, p, size, ps, pc);
            recvcounts[p] = pc * len;
        }
        displs[0] = 0;
        for (int p = 1; p < size; ++p) {
            displs[p] = displs[p - 1] + recvcounts[p - 1];
        }
        int total_recv = 0;
        for (int p = 0; p < size; ++p) total_recv += recvcounts[p];
        recvbuf.resize(total_recv);

        // 用 Allgatherv 交换所有进程打包后的片段
        MPI_Allgatherv(
            sendbuf.data(), send_elems, MPI_INT,
            recvbuf.data(), recvcounts.data(), displs.data(), MPI_INT,
            MPI_COMM_WORLD
        );

        // 将 recvbuf 解包回 f 的对应段位置
        {
            int offset = 0;
            for (int p = 0; p < size; ++p) {
                int ps, pc;
                block_partition(total_segments, p, size, ps, pc);
                for (int k = 0; k < pc; ++k) {
                    int seg_idx = ps + k;
                    int st = seg_idx * len;
                    std::memcpy(f + st, &recvbuf[offset + k * len], len * sizeof(int));
                }
                offset += pc * len;
            }
        }
    }

    // 最终的归一化和位反转（只在主进程执行）
    if (rank == 0) {
        const ll invl = fpow(lim, P - 2, P);
        for (int i = 0; i < lim; ++i) {
            f[i] = (invl * f[i]) % P;
        }
        std::reverse(f + 1, f + lim);
    }

    // 广播最终结果
    MPI_Bcast(f, lim, MPI_INT, 0, MPI_COMM_WORLD);
}

/**
 * 使用 NTT 进行多项式乘法的优化实现
 * @param a_input_from_main 多项式 A 系数数组
 * @param b_input_from_main 多项式 B 系数数组
 * @param ab_output_from_main 输出结果数组
 * @param n 多项式长度
 * @param p 模数
 * @param gen 原根
 * @param rank MPI 进程 ID
 * @param size MPI 进程总数
 */
void poly_multiply_optimized(int *a_input_from_main, int *b_input_from_main, int *ab_output_from_main,
                             int n, int p, int gen, int rank, int size) {
    if (n == 0) {
        if (rank == 0 && ab_output_from_main != nullptr) {
            // nothing
        }
        return;
    }

    int l = 0;
    while ((1 << l) < (n == 1 ? 2 : (2 * n))) {
        l++;
        if (l > 25 && rank == 0) {
            std::cerr << "Warning: l is very large: " << l << " for n=" << n << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    int lim = 1 << l;

    std::vector<int> A_vec(lim, 0);
    std::vector<int> B_vec(lim, 0);
    std::vector<int> w_vec(lim, 0);

    if (rank == 0) {
        if (n > 0) {
            memcpy(A_vec.data(), a_input_from_main, n * sizeof(int));
            memcpy(B_vec.data(), b_input_from_main, n * sizeof(int));
        }
    }

    MPI_Bcast(A_vec.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B_vec.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

    calc_powg(w_vec.data(), lim, p, gen);

    // 正变换
    DIF(A_vec.data(), l, p, w_vec.data(), rank, size);
    DIF(B_vec.data(), l, p, w_vec.data(), rank, size);

    // 并行化点乘操作
    for (int i = rank; i < lim; i += size) {
        A_vec[i] = (ll)A_vec[i] * B_vec[i] % p;
    }

    // 将点乘结果合并到所有进程的 A_vec 中
    {
        int send_elems = (lim + size - 1) / size; // 每个进程分配 ceil(lim/size) 个元素
        // 但事实上我们只是使用稀疏存储，不需要全部元素，都由块分配函数处理
        // 这里直接使用 MPI_Allgather，以最简单方式交换每个索引
        std::vector<int> sendbuf(lim, 0), recvbuf(lim, 0);
        for (int i = rank; i < lim; i += size) {
            sendbuf[i] = A_vec[i];
        }
        MPI_Allreduce(sendbuf.data(), recvbuf.data(), lim, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        std::memcpy(A_vec.data(), recvbuf.data(), lim * sizeof(int));
    }

    // 逆变换
    DIT(A_vec.data(), l, p, w_vec.data(), rank, size);

    if (rank == 0) {
        int result_len = 2 * n - 1;
        if (result_len < 0) result_len = 0;

        if (ab_output_from_main != nullptr && result_len > 0) {
            memcpy(ab_output_from_main, A_vec.data(), std::min(result_len, lim) * sizeof(int));
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int MAX_N_EXPECTED = 1 << 18;
    std::vector<int> main_a(MAX_N_EXPECTED);
    std::vector<int> main_b(MAX_N_EXPECTED);
    std::vector<int> main_ab(2 * MAX_N_EXPECTED);

    int test_begin = 0;
    int test_end = 3;
    for (int i = test_begin; i <= test_end; ++i) {
        int n_val = 0, p_val = 0;

        if (rank == 0) {
            std::string str1 = "/nttdata/";
            std::string str2 = std::to_string(i);
            std::string strin = str1 + str2 + ".in";
            std::ifstream fin(strin);
            if (!fin) {
                std::cerr << "Rank 0: 无法打开文件 " << strin << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            fin >> n_val >> p_val;
            if (n_val > MAX_N_EXPECTED) {
                std::cerr << "Rank 0: Input n_val " << n_val
                          << " for id " << i << " exceeds MAX_N_EXPECTED " << MAX_N_EXPECTED << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (n_val < 0) {
                std::cerr << "Rank 0: Input n_val " << n_val << " is negative for id " << i << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            for (int k = 0; k < n_val; ++k) {
                if (!(fin >> main_a[k])) {
                    std::cerr << "Rank 0: Error reading a[" << k << "] for id " << i << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
            for (int k = 0; k < n_val; ++k) {
                if (!(fin >> main_b[k])) {
                    std::cerr << "Rank 0: Error reading b[" << k << "] for id " << i << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
            fin.close();
        }

        MPI_Bcast(&n_val, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p_val, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (n_val <= 0) {
            if (rank == 0) {
                fWrite(main_ab.data(), n_val, i);
                std::cout << "n = " << n_val << " p = " << p_val
                          << " for id " << i << ". Skipping computation due to n <= 0." << std::endl;
            }
            continue;
        }

        std::fill(main_ab.begin(), main_ab.begin() + (2 * n_val > 0 ? 2 * n_val - 1 : 0), 0);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        poly_multiply_optimized(main_a.data(), main_b.data(), main_ab.data(), n_val, p_val, 3, rank, size);

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (rank == 0) {
            double us = (t1 - t0) * 1e6;
            fWrite(main_ab.data(), n_val, i);
            fCheck(main_ab.data(), n_val, i, rank);
            std::cout << "average latency for n=" << n_val << " p=" << p_val
                      << " : " << us << " us" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
