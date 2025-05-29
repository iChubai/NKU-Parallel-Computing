/*
 * ===========================================
 * 文件名: main_pthread_DIF_DIT.cc
 * 描述: 使用DIF DIT实现NTT的pthread并行算法
 * ===========================================
 */


#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <algorithm>
#include <pthread.h>
#include <vector>
#include <cstdlib>

typedef long long ll;

static const int NUM_THREADS = 4;

// fRead: Reads polynomial coefficients, degree, and modulus from input file.
// Args: a(int*)-coeff array A, b(int*)-coeff array B, n(int*)-degree, p(int*)-modulus, input_id(int)-test case ID.
void fRead(int *a, int *b, int *n, int *p, int input_id) {
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);

    if (!fin.is_open()) {
        std::cerr << "Error: Could not open input file: " << data_path << std::endl;
        exit(EXIT_FAILURE);
    }

    fin >> *n >> *p;

    if (*p == 0) {
        std::cerr << "Error: Modulus p read from file " << data_path << " is zero." << std::endl;
        fin.close();
        exit(EXIT_FAILURE);
    }
    if (*n == 0) {
        std::cerr << "Error: Degree n read from file " << data_path << " is zero." << std::endl;
        fin.close();
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < *n; i++){
        fin>>a[i];
    }
    for (int i = 0; i < *n; i++){
        fin>>b[i];
    }
}

// fCheck: Checks computed polynomial product against expected output file.
// Args: ab(int*)-computed product, n(int)-original degree, input_id(int)-test case ID.
void fCheck(int *ab, int n, int input_id) {
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++){
        int x;
        fin>>x;
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

// fWrite: Writes polynomial product to an output file.
// Args: ab(int*)-product to write, n(int)-original degree, input_id(int)-test case ID.
void fWrite(int *ab, int n, int input_id) {
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++){
        fout<<ab[i]<<'\n';
    }
}

// fpow: Computes (a^b) % P.
// Args: a(ll)-base, b(ll)-exponent, P(int)-modulus. Ret: (ll) result.
inline ll fpow(ll a, ll b, int P) {
    ll res = 1; a %= P;
    for (; b; b >>= 1) {
        if (b & 1) (res *= a) %= P;
        (a *= a) %= P;
    }
    return res;
}

// calc_powg: Precomputes twiddle factors for NTT.
// Args: w(int*)-output array for twiddles, G(int)-transform size, P(int)-modulus, gen(int)-primitive root.
void calc_powg(int w[], int G, int P, int gen) {
    w[0] = 1; ll f;
    const int g = fpow(gen, (P-1)/G, P);
    for (int t = 0; (1<<(t+1)) < G; ++t) {
        f = w[1<<t] = fpow(g, G>>(t+2), P);
        for (int x = 1<<t; x < 1<<(t+1); ++x)
            w[x] = (ll)f * w[x - (1<<t)] % P;
    }
}

typedef struct {
    int* f;
    int start_block_idx;
    int end_block_idx;
    int len;
    int P;
    int* w_factors;
} DIF_DIT_Args;

void* dif_stage_thread(void* arg) {
    DIF_DIT_Args* args = (DIF_DIT_Args*)arg;
    for (int t = args->start_block_idx; t < args->end_block_idx; ++t) {
        int st = t * args->len;
        for (int i = st; i < st + args->len / 2; ++i) {
            ll g = args->f[i];
            ll h = (ll)args->f[i + args->len / 2] * args->w_factors[t] % args->P;
            args->f[i] = (g + h) % args->P;
            args->f[i + args->len / 2] = (args->P + g - h) % args->P;
        }
    }
    pthread_exit(NULL);
}

// DIF: Parallel Decimation-In-Frequency NTT (forward transform).
// Args: f(int*)-array, l(int)-log2(lim), P(int)-modulus, w_factors(int*)-twiddle factors.
void DIF(int f[], int l, int P, int w_factors[]) {
    int lim = 1 << l;
    pthread_t threads[NUM_THREADS];
    DIF_DIT_Args thread_args[NUM_THREADS];

    for (int len = lim; len > 1; len >>= 1) {
        int num_total_blocks = lim / len;
        int blocks_per_thread = (num_total_blocks + NUM_THREADS - 1) / NUM_THREADS;

        for (int i = 0; i < NUM_THREADS; ++i) {
            thread_args[i].f = f;
            thread_args[i].start_block_idx = i * blocks_per_thread;
            thread_args[i].end_block_idx = std::min((i + 1) * blocks_per_thread, num_total_blocks);
            thread_args[i].len = len;
            thread_args[i].P = P;
            thread_args[i].w_factors = w_factors;

            if (thread_args[i].start_block_idx < thread_args[i].end_block_idx) {
                pthread_create(&threads[i], NULL, dif_stage_thread, &thread_args[i]);
            }
        }

        for (int i = 0; i < NUM_THREADS; ++i) {
            if (thread_args[i].start_block_idx < thread_args[i].end_block_idx) {
                pthread_join(threads[i], NULL);
            }
        }
    }
}

void* dit_stage_thread(void* arg) {
    DIF_DIT_Args* args = (DIF_DIT_Args*)arg;
    for (int t = args->start_block_idx; t < args->end_block_idx; ++t) {
        int st = t * args->len;
        for (int i = st; i < st + args->len / 2; ++i) {
            ll g = args->f[i];
            ll h = args->f[i + args->len / 2];
            args->f[i] = (g + h) % args->P;
            args->f[i + args->len / 2] = (args->P + g - h) * (ll)args->w_factors[t] % args->P;
        }
    }
    pthread_exit(NULL);
}

typedef struct {
    int* f;
    int start_idx;
    int end_idx;
    ll invl;
    int P;
} Scale_Args;

void* scale_thread(void* arg) {
    Scale_Args* args = (Scale_Args*)arg;
    for (int i = args->start_idx; i < args->end_idx; ++i) {
        args->f[i] = args->invl * args->f[i] % args->P;
    }
    pthread_exit(NULL);
}

// DIT: Parallel Decimation-In-Time NTT (inverse transform).
// Args: f(int*)-array, l(int)-log2(lim), P(int)-modulus, w_factors(int*)-twiddle factors.
void DIT(int f[], int l, int P, int w_factors[]) {
    int lim = 1 << l;
    pthread_t threads[NUM_THREADS];
    DIF_DIT_Args dit_thread_args[NUM_THREADS];

    for (int len = 2; len <= lim; len <<= 1) {
        int num_total_blocks = lim / len;
        int blocks_per_thread = (num_total_blocks + NUM_THREADS - 1) / NUM_THREADS;

        for (int i = 0; i < NUM_THREADS; ++i) {
            dit_thread_args[i].f = f;
            dit_thread_args[i].start_block_idx = i * blocks_per_thread;
            dit_thread_args[i].end_block_idx = std::min((i + 1) * blocks_per_thread, num_total_blocks);
            dit_thread_args[i].len = len;
            dit_thread_args[i].P = P;
            dit_thread_args[i].w_factors = w_factors;

            if (dit_thread_args[i].start_block_idx < dit_thread_args[i].end_block_idx) {
                pthread_create(&threads[i], NULL, dit_stage_thread, &dit_thread_args[i]);
            }
        }
        for (int i = 0; i < NUM_THREADS; ++i) {
            if (dit_thread_args[i].start_block_idx < dit_thread_args[i].end_block_idx) {
                pthread_join(threads[i], NULL);
            }
        }
    }

    const ll invl = fpow(lim, P - 2, P);
    Scale_Args scale_args[NUM_THREADS];
    int elements_per_thread_scale = (lim + NUM_THREADS - 1) / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; ++i) {
        scale_args[i].f = f;
        scale_args[i].start_idx = i * elements_per_thread_scale;
        scale_args[i].end_idx = std::min((i + 1) * elements_per_thread_scale, lim);
        scale_args[i].invl = invl;
        scale_args[i].P = P;
        if (scale_args[i].start_idx < scale_args[i].end_idx) {
            pthread_create(&threads[i], NULL, scale_thread, &scale_args[i]);
        }
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
        if (scale_args[i].start_idx < scale_args[i].end_idx) {
            pthread_join(threads[i], NULL);
        }
    }
    std::reverse(f + 1, f + lim);
}

typedef struct {
    int* A;
    int* B;
    int start_idx;
    int end_idx;
    int p;
} PointwiseMul_Args;

void* pointwise_mul_thread(void* arg) {
    PointwiseMul_Args* args = (PointwiseMul_Args*)arg;
    for (int i = args->start_idx; i < args->end_idx; ++i) {
        args->A[i] = (ll)args->A[i] * args->B[i] % args->p;
    }
    pthread_exit(NULL);
}

// poly_multiply_optimized: Parallel polynomial multiplication using DIF/DIT NTT.
// Args: a(int*)-poly A, b(int*)-poly B, ab(int*)-result poly, n(int)-degree, p(int)-modulus, gen(int)-primitive root.
void poly_multiply_optimized(int *a, int *b, int *ab, int n, int p, int gen = 3) {
    memset(ab, 0, sizeof(int) * (2 * n - 1));
    int l = 0;
    while ((1 << l) < 2 * n) l++;
    int lim = 1 << l;
    
    int *A = new int[lim]();
    int *B = new int[lim]();
    int *w = new int[lim]();
    
    for (int i = 0; i < n; ++i) {
        A[i] = a[i];
        B[i] = b[i];
    }
    
    calc_powg(w, lim, p, gen);
    DIF(A, l, p, w);
    DIF(B, l, p, w);
    
    pthread_t threads_pm[NUM_THREADS];
    PointwiseMul_Args pm_args[NUM_THREADS];
    int elements_per_thread_pm = (lim + NUM_THREADS - 1) / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; ++i) {
        pm_args[i].A = A;
        pm_args[i].B = B;
        pm_args[i].start_idx = i * elements_per_thread_pm;
        pm_args[i].end_idx = std::min((i + 1) * elements_per_thread_pm, lim);
        pm_args[i].p = p;
        if (pm_args[i].start_idx < pm_args[i].end_idx) {
            pthread_create(&threads_pm[i], NULL, pointwise_mul_thread, &pm_args[i]);
        }
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
        if (pm_args[i].start_idx < pm_args[i].end_idx) {
            pthread_join(threads_pm[i], NULL);
        }
    }
    
    DIT(A, l, p, w);
    
    for (int i = 0; i < 2 * n - 1; ++i)
        ab[i] = A[i];
    
    delete[] A;
    delete[] B;
    delete[] w;
}

int a[300000], b[300000], ab[300000];
// main: Runs NTT tests for different inputs and parameters.
// Args: argc(int)-arg count, argv(char*[])-arg values.
int main(int argc, char *argv[])
{
    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入模数分别为 7340033 104857601 469762049 263882790666241
    // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
    // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1
    int test_begin = 0;
    int test_end = 3;
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();
        
        // 使用优化的多项式乘法
        poly_multiply_optimized(a, b, ab, n_, p_);
        
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<std::endl;
        fWrite(ab, n_, i);
    }
    return 0;
}
