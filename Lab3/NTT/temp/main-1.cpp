#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>

#include <omp.h>   // OpenMP
// Conditionally include MPI
#ifdef USE_MPI
#include <mpi.h>   // MPI
#endif
#include <pthread.h> // For pthreads
// #include <immintrin.h> // For AVX intrinsics -- Temporarily removed
#include <thread>      // For std::thread::hardware_concurrency

/* =============================== 1. 通用工具 =============================== */
class Timer {
private:
    using clk = std::chrono::high_resolution_clock;
    std::string tag;
    clk::time_point beg;
    std::vector<double> times;  // 存储多次计时结果
    bool is_running = false;

    static std::string format_time(double seconds) {
        if (seconds < 1e-6) return std::to_string(seconds * 1e9) + " ns";
        if (seconds < 1e-3) return std::to_string(seconds * 1e6) + " μs";
        if (seconds < 1.0) return std::to_string(seconds * 1e3) + " ms";
        return std::to_string(seconds) + " s";
    }

public:
    explicit Timer(std::string t) : tag(std::move(t)) {}
    
    void start() {
        if (!is_running) {
            beg = clk::now();
            is_running = true;
        }
    }
    
    void stop() {
        if (is_running) {
            auto end = clk::now();
            double sec = std::chrono::duration<double>(end - beg).count();
            times.push_back(sec);
            is_running = false;
        }
    }
    
    void reset() {
        times.clear();
        is_running = false;
    }
    
    void print_stats() const {
        if (times.empty()) {
            std::cout << "[" << tag << "] No timing data available\n";
            return;
        }
        
        double total = 0.0;
        double min_time = times[0];
        double max_time = times[0];
        
        for (double t : times) {
            total += t;
            min_time = std::min(min_time, t);
            max_time = std::max(max_time, t);
        }
        
        double avg_time = total / times.size();
        
        std::cout << "[" << tag << "] Statistics:\n"
                  << "  运行次数 : " << times.size() << " 次\n"
                  << "  总时间   : " << format_time(total) << "\n"
                  << "  平均时间 : " << format_time(avg_time) << "\n"
                  << "  最短时间 : " << format_time(min_time) << "\n"
                  << "  最长时间 : " << format_time(max_time) << "\n";
    }
    
    ~Timer() {
        if (is_running) stop();
        print_stats();
    }
};

inline double gflops(double sec, long long N, long long M, long long P)
{
    return 2.0 * N * M * P / 1e9 / sec;
}

void init_matrix(std::vector<double>& mat, int rows, int cols)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (size_t i = 0; i < mat.size(); ++i) mat[i] = dist(gen);
}

bool validate(const std::vector<double>& A,
              const std::vector<double>& B,
              int rows, int cols,
              double tol = 1e-6)
{
    for (int i = 0; i < rows * cols; ++i)
        if (std::abs(A[i] - B[i]) > tol) return false;
    return true;
}

/* ========================= 2. 各种矩阵乘实现 ========================= */
void matmul_baseline(const std::vector<double>& A,
                     const std::vector<double>& B,
                     std::vector<double>& C,
                     int N, int M, int P)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
}

/* --- 2.1 OpenMP 并行 --- */
void matmul_openmp(const std::vector<double>& A,
                   const std::vector<double>& B,
                   std::vector<double>& C,
                   int N, int M, int P)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
}

void matmul_openmp_v2(const std::vector<double>& A,
                   const std::vector<double>& B,
                   std::vector<double>& C,
                   int N, int M, int P)
{    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < M; ++k) {
            double a_ik = A[i*M + k];
            for (int j = 0; j < P; ++j) {
                C[i*P + j] += a_ik * B[k*P + j];
            }
        }
    }
}

/* --- 2.2 Cache‑friendly Block Tiling + OpenMP --- */
void matmul_block_tiling(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C,
                         int N, int M, int P,
                         int BS = 64)              // block_size
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += BS)
        for (int jj = 0; jj < P; jj += BS)
            for (int kk = 0; kk < M; kk += BS)
                for (int i = ii; i < std::min(ii + BS, N); ++i)
                    for (int j = jj; j < std::min(jj + BS, P); ++j) {
                        double sum = C[i * P + j];
                        for (int k = kk; k < std::min(kk + BS, M); ++k)
                            sum += A[i * M + k] * B[k * P + j];
                        C[i * P + j] = sum;
                    }
}

/* --- 2.3 MPI 行划分实现 ---
 *     每个 rank 负责 N_local = N/size 行的 A 与 C。
 *     简单起见，此处假设 N 可被 size 整除且矩阵尺寸不超过内存。
 */
void matmul_mpi(int N, int M, int P)
{
#ifdef USE_MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    /* 行数均分 */
    if (N % size != 0) {
        if (rank==0) std::cerr << "N must be divisible by #ranks.\n";
        return;
    }
    int Nloc = N / size;

    /* 仅根进程生成完整矩阵 */
    std::vector<double> A(Nloc * M);
    std::vector<double> B(M * P);
    std::vector<double> Cloc(Nloc * P, 0), C;      // root later gathers

    if (rank == 0)
    {
        std::vector<double> Afull(N * M);
        init_matrix(Afull,N,M);
        init_matrix(B,M,P);

        /* broadcast B */
        MPI_Bcast(B.data(), M*P, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        /* scatter A */
        MPI_Scatter(Afull.data(), Nloc*M, MPI_DOUBLE,
                    A.data(),     Nloc*M, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Bcast(B.data(), M*P, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(nullptr,0,MPI_DOUBLE,
                    A.data(), Nloc*M,MPI_DOUBLE,
                    0,MPI_COMM_WORLD);
    }

    /* 计时 */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* 本地 GEMM */
    for (int i = 0; i < Nloc; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A[i*M+k]*B[k*P+j];
            Cloc[i*P+j]=sum;
        }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local = t1 - t0, global;

    MPI_Reduce(&local,&global,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    /* gather 结果到 root 便于验证 */
    if (rank==0) C.resize(N*P);
    MPI_Gather(Cloc.data(), Nloc*P, MPI_DOUBLE,
               C.data(),    Nloc*P, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank==0)
        std::cout << "[MPI] Time(max) = " << global
                  << " s,  GFLOPS ≈ "
                  << gflops(global,N,M,P) << '\n';
#else
    // Provide a stub or error if MPI is called without being compiled
    if (N > 0) { // Avoid unused parameter warnings if N,M,P are 0
        std::cerr << "MPI support not compiled. Recompile with -DUSE_MPI and MPI libraries." << std::endl;
    }
#endif
}
void matmul_mpi_nb(int N, int M, int P) {
#ifdef USE_MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) std::cerr << "N must be divisible by #ranks.\n";
        return;
    }
    int Nloc = N / size;

    // 本地缓冲区
    std::vector<double> A(Nloc * M);
    std::vector<double> B(M * P);
    std::vector<double> Cloc(Nloc * P, 0.0);
    std::vector<double> C;  // 仅 root 用于收集

    // 仅 root 生成完整矩阵
    std::vector<double> Afull;
    if (rank == 0) {
        Afull.resize(N * M);
        init_matrix(Afull,N,M);
        init_matrix(B,M,P); 
    }

    // 非阻塞广播 B，非阻塞 Scatter A
    MPI_Request reqs[2];
    MPI_Ibcast(B.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD, &reqs[0]);

    MPI_Iscatter(
        rank == 0 ? Afull.data() : nullptr, // sendbuf (只有 root 发送)
        Nloc * M, MPI_DOUBLE,
        A.data(),    // recvbuf
        Nloc * M, MPI_DOUBLE,
        0, MPI_COMM_WORLD,
        &reqs[1]
    );

    // 等待 B 和 A 分发完成后再进行本地计算
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    // 计时开始
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // 本地 GEMM（可选再内嵌 Cache-Blocking 或 OpenMP）
    for (int i = 0; i < Nloc; ++i) {
        for (int k = 0; k < M; ++k) {
            double a_ik = A[i * M + k];
            for (int j = 0; j < P; ++j) {
                Cloc[i * P + j] += a_ik * B[k * P + j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_t = t1 - t0, max_t;

    MPI_Reduce(&local_t, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // 非阻塞收集 Cloc 到 root 的 C
    if (rank == 0) C.resize(N * P);
    MPI_Igather(
        Cloc.data(), Nloc * P, MPI_DOUBLE,
        rank == 0 ? C.data() : nullptr, // recvbuf
        Nloc * P, MPI_DOUBLE,
        0, MPI_COMM_WORLD,
        &reqs[0]
    );

    // 等待收集完成
    MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);

    if (rank == 0) {
        std::cout << "[MPI-NB] Time(max) = " << max_t
                  << " s,  GFLOPS ≈ " << gflops(max_t, N, M, P) << "\n";
    }
#else
    // Provide a stub or error if MPI is called without being compiled
    if (N > 0) { // Avoid unused parameter warnings
        std::cerr << "MPI support not compiled. Recompile with -DUSE_MPI and MPI libraries." << std::endl;
    }
#endif
}

// Define a struct to pass data to each thread
typedef struct {
    const std::vector<double>* A;
    const std::vector<double>* B;
    std::vector<double>* C;
    int N, M, P;
    int start_row;
    int end_row;
} pthread_matmul_args_t;

// Thread worker function (scalar version)
void* thread_worker_matmul(void* args_ptr) {
    pthread_matmul_args_t* args = (pthread_matmul_args_t*)args_ptr;

    const double* A_ptr = args->A->data();
    const double* B_ptr = args->B->data();
    double* C_ptr = args->C->data();
    int M = args->M;
    int P = args->P;

    for (int i = args->start_row; i < args->end_row; ++i) {
        for (int k = 0; k < M; ++k) {
            double a_ik = A_ptr[i * M + k];
            for (int j = 0; j < P; ++j) {
                C_ptr[i * P + j] += a_ik * B_ptr[k * P + j];
            }
        }
    }
    pthread_exit(NULL);
}

/* --- 2.4 预留自定义方法 --- */
void matmul_other(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C,
                  int N, int M, int P)
{
    // pthreads with manual AVX SIMD
    // Determine number of threads - e.g., number of hardware cores
    // For simplicity and to avoid oversubscription if OpenMP is used elsewhere,
    // let's cap it or use a fixed number. std::thread::hardware_concurrency() is a good start.
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Fallback if detection fails
    // num_threads = 4; // Or simply fix it for testing

    std::vector<pthread_t> threads(num_threads);
    std::vector<pthread_matmul_args_t> thread_args(num_threads);

    int rows_per_thread = N / num_threads;
    int remaining_rows = N % num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        thread_args[t].A = &A;
        thread_args[t].B = &B;
        thread_args[t].C = &C;
        thread_args[t].N = N;
        thread_args[t].M = M;
        thread_args[t].P = P;
        
        thread_args[t].start_row = t * rows_per_thread;
        thread_args[t].end_row = (t + 1) * rows_per_thread;

        if (t == num_threads - 1) {
            thread_args[t].end_row += remaining_rows; // Add remaining rows to the last thread
        }
        // Ensure end_row does not exceed N
        if (thread_args[t].end_row > N) thread_args[t].end_row = N;
        
        // Ensure start_row is not beyond end_row if N is small
        if (thread_args[t].start_row >= thread_args[t].end_row && N > 0) {
            if (t > 0 && thread_args[t-1].end_row < N ) { // if previous thread didn't cover all
                 thread_args[t].start_row = thread_args[t-1].end_row; // start where previous ended
                 thread_args[t].end_row = N; // take all remaining
            } else if (t==0) { // if first thread and N is small
                 thread_args[t].start_row = 0;
                 thread_args[t].end_row = N;
            } else { // if this thread has no work
                thread_args[t].start_row = N; // Mark as no work
                thread_args[t].end_row = N;
            }
        }


        if (thread_args[t].start_row < thread_args[t].end_row) { // Only create thread if there's work
            int rc = pthread_create(&threads[t], NULL, thread_worker_matmul, (void*)&thread_args[t]);
            if (rc) {
                std::cerr << "Error:unable to create thread," << rc << std::endl;
                exit(-1);
            }
        }
    }

    // Wait for all threads to complete
    for (unsigned int t = 0; t < num_threads; ++t) {
        if (thread_args[t].start_row < thread_args[t].end_row) { // Only join if thread was created
            pthread_join(threads[t], NULL);
        }
    }
}

/* ================================ 3. main ================================ */
int main(int argc, char** argv)
{
    const int N = 1024, M = 2048, P = 512;  // 可自行修改
    const int RUNS = 5;  // 指定运行次数
    std::string mode = (argc >= 2) ? argv[1] : "baseline";

    /* ==== MPI 独占分支 ==== */
    if (mode == "mpi")
    {
#ifdef USE_MPI
        MPI_Init(&argc,&argv);
        matmul_mpi_nb(N,M,P); // Or matmul_mpi(N,M,P) if you prefer the blocking one
        MPI_Finalize();
        return 0;
#else
        std::cerr << "MPI mode selected, but MPI support not compiled. Recompile with -DUSE_MPI and MPI libraries." << std::endl;
        return 1;
#endif
    }

    /* ==== 单进程模式 ==== */
    std::vector<double> A(N*M), B(M*P), C(N*P,0), Cref(N*P,0);
    init_matrix(A,N,M);
    init_matrix(B,M,P);

    // 先运行一次baseline作为参考
    {   
        Timer t("Baseline");
        t.start();
        matmul_baseline(A,B,Cref,N,M,P);
        t.stop();
    }

    if (mode=="baseline")
    {
        // baseline 计时已在上方完成
        std::cout << "[Baseline] DONE  (参考)\n";
    }
    else if (mode=="openmp")
    {
        Timer t("OpenMP_v2");
        for(int run = 0; run < RUNS; ++run) {
            std::fill(C.begin(),C.end(),0);
            t.start();
            matmul_openmp_v2(A,B,C,N,M,P);
            t.stop();
        }
        std::cout << "[OpenMP]   Valid : " << std::boolalpha
                  << validate(C,Cref,N,P) << '\n';
    }
    else if (mode=="block")
    {
        Timer t("Block‑Tiling");
        for(int run = 0; run < RUNS; ++run) {
            std::fill(C.begin(),C.end(),0);
            t.start();
            matmul_block_tiling(A,B,C,N,M,P,64);
            t.stop();
        }
        std::cout << "[Block]    Valid : " << std::boolalpha
                  << validate(C,Cref,N,P) << '\n';
    }
    else if (mode=="other")
    {
        Timer t("Other‑Method");
        for(int run = 0; run < RUNS; ++run) {
            std::fill(C.begin(),C.end(),0);
            t.start();
            matmul_other(A,B,C,N,M,P);
            t.stop();
        }
        std::cout << "[Other]    Valid : " << std::boolalpha
                  << validate(C,Cref,N,P) << '\n';
    }
    else
    {
        std::cerr << "Usage: ./matmul [baseline|openmp|block|mpi|other]\n";
    }
    return 0;
}
