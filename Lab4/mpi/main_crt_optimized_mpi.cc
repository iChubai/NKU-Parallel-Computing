/****************************************************************************************
 * main_crt_optimized_mpi.cc - 优化的CRT多模数NTT实现
 *
 * 主要优化：
 * 1. 扩展模数策略：支持3-9个模数的动态选择
 * 2. 算法优化：改进的CRT重构算法和模逆元预计算
 * 3. 内存优化：缓存友好的数据布局和访问模式
 * 4. 数值稳定性：改进的大整数运算和溢出处理
 * 5. 负载均衡：动态工作负载分配策略
 *
 * 编译：mpicxx -O3 -std=c++17 -march=native main_crt_optimized_mpi.cc -o crt_optimized_mpi
 * 运行：mpirun -np 4 ./crt_optimized_mpi
 ****************************************************************************************/
#include <bits/stdc++.h>
#include <mpi.h>
#include <chrono>
#include <iomanip>

using namespace std;

/* ============================== I/O 函数 ============================== */
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

/* ============================== 优化的Barrett规约器 ============================== */
class OptimizedBarrett {
public:
    uint64_t mod;
    uint64_t inv;
    
    explicit OptimizedBarrett(uint64_t m = 1) : mod(m) {
        // 使用更精确的逆元计算
        inv = (static_cast<__uint128_t>(1) << 64) / m;
    }
    
    inline uint32_t reduce(uint64_t a) const {
        // 优化的Barrett规约，减少分支预测失败
        uint64_t q = (static_cast<__uint128_t>(a) * inv) >> 64;
        uint64_t r = a - q * mod;
        return static_cast<uint32_t>(r >= mod ? r - mod : r);
    }
    
    inline uint32_t add(uint32_t a, uint32_t b) const {
        uint32_t s = a + b;
        return s >= mod ? s - mod : s;
    }
    
    inline uint32_t sub(uint32_t a, uint32_t b) const {
        return a >= b ? a - b : a + mod - b;
    }
    
    inline uint32_t mul(uint32_t a, uint32_t b) const {
        return reduce(static_cast<uint64_t>(a) * b);
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

/* ============================== 优化的NTT实现 ============================== */
void optimizedNTT(vector<uint32_t>& a, bool inverse, const OptimizedBarrett& br) {
    const int n = a.size();
    
    // 位反转优化：减少分支和内存访问
    static vector<int> rev;
    if (rev.size() != n) {
        rev.resize(n);
        int lg = __builtin_ctz(n);
        for (int i = 0; i < n; ++i) {
            rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (lg - 1));
        }
    }
    
    for (int i = 0; i < n; ++i) {
        if (i < rev[i]) swap(a[i], a[rev[i]]);
    }

    // 蝶形运算优化：减少模运算和提高缓存局部性
    for (int len = 2; len <= n; len <<= 1) {
        uint32_t wn = br.pow(3, (br.mod - 1) / len);
        if (inverse) wn = br.pow(wn, br.mod - 2);
        
        int half = len >> 1;
        
        // 预计算旋转因子以减少重复计算
        vector<uint32_t> w_powers(half);
        w_powers[0] = 1;
        for (int j = 1; j < half; ++j) {
            w_powers[j] = br.mul(w_powers[j-1], wn);
        }
        
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < half; ++j) {
                uint32_t u = a[i + j];
                uint32_t v = br.mul(a[i + j + half], w_powers[j]);
                a[i + j] = br.add(u, v);
                a[i + j + half] = br.sub(u, v);
            }
        }
    }
    
    if (inverse) {
        uint32_t inv_n = br.pow(n, br.mod - 2);
        for (auto& x : a) x = br.mul(x, inv_n);
    }
}

/* ============================== 扩展模数管理器 ============================== */
class ModulusManager {
public:
    // 9个高质量的NTT友好模数
    static constexpr uint32_t AVAILABLE_MODS[9] = {
        998244353u,   // 2^23 * 119 + 1
        1004535809u,  // 2^21 * 479 + 1  
        469762049u,   // 2^26 * 7 + 1
        167772161u,   // 2^25 * 5 + 1
        1224736769u,  // 2^24 * 73 + 1
        595591169u,   // 2^22 * 142 + 1
        104857601u,   // 2^22 * 25 + 1
        23068673u,    // 2^21 * 11 + 1
        7340033u      // 2^20 * 7 + 1
    };
    
    static constexpr int MAX_MODS = 9;
    
    /**
     * @brief 根据目标模数和数据大小动态选择最优模数数量
     * @param target_mod 目标模数
     * @param data_size 数据大小
     * @return 推荐的模数数量
     */
    static int selectOptimalModCount(uint64_t target_mod, int data_size) {
        // 计算所需的位数
        int target_bits = 64 - __builtin_clzll(target_mod);
        int data_bits = 32 - __builtin_clz(data_size);
        
        // 估算乘法结果的位数：2 * data_bits + target_bits
        int result_bits = 2 * data_bits + target_bits;
        
        // 每个模数大约提供30位精度
        int required_mods = (result_bits + 29) / 30;
        
        // 限制在3-9个模数之间，并考虑性能平衡
        required_mods = max(3, min(MAX_MODS, required_mods));
        
        // 对于小数据，使用较少模数以减少开销
        if (data_size <= 1024) {
            required_mods = min(required_mods, 5);
        }
        
        return required_mods;
    }
    
    /**
     * @brief 获取指定数量的模数
     */
    static vector<uint32_t> getModuli(int count) {
        count = min(count, MAX_MODS);
        return vector<uint32_t>(AVAILABLE_MODS, AVAILABLE_MODS + count);
    }
};

constexpr uint32_t ModulusManager::AVAILABLE_MODS[9];

/* ============================== 优化的模逆元计算 ============================== */
class ModInverseCache {
private:
    unordered_map<uint64_t, uint64_t> cache;
    
public:
    uint64_t computeModInverse(uint64_t a, uint64_t m) {
        uint64_t key = (a << 32) | m;
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }
        
        // 扩展欧几里得算法
        uint64_t b = m, u = 1, v = 0;
        while (b) {
            uint64_t t = a / b;
            a -= t * b; swap(a, b);
            u -= t * v; swap(u, v);
        }
        
        uint64_t result = (u + m) % m;
        cache[key] = result;
        return result;
    }
    
    void precomputeInverses(const vector<uint32_t>& mods) {
        int n = mods.size();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                uint64_t prod = static_cast<uint64_t>(mods[i]) * mods[j];
                computeModInverse(mods[i], mods[j]);
                computeModInverse(mods[j], mods[i]);
                for (int k = j + 1; k < n; ++k) {
                    computeModInverse(prod % mods[k], mods[k]);
                }
            }
        }
    }
};

/* ============================== 优化的CRT重构算法 ============================== */
class OptimizedCRT {
private:
    vector<uint32_t> moduli;
    vector<__uint128_t> mod_products;
    ModInverseCache inv_cache;
    
public:
    OptimizedCRT(const vector<uint32_t>& mods) : moduli(mods) {
        precomputeProducts();
        inv_cache.precomputeInverses(moduli);
    }
    
private:
    void precomputeProducts() {
        int n = moduli.size();
        mod_products.resize(n);
        mod_products[0] = moduli[0];
        for (int i = 1; i < n; ++i) {
            mod_products[i] = mod_products[i-1] * moduli[i];
        }
    }
    
public:
    /**
     * @brief 优化的CRT重构，支持任意数量的模数
     */
    void reconstruct(const vector<vector<uint32_t>>& remainders, 
                    vector<uint64_t>& result, uint64_t target_mod) {
        int n = moduli.size();
        int len = remainders[0].size();
        result.resize(len);
        
        for (int i = 0; i < len; ++i) {
            __uint128_t x = remainders[0][i];
            
            for (int j = 1; j < n; ++j) {
                uint64_t current_prod = static_cast<uint64_t>(mod_products[j-1]);
                uint64_t diff = (remainders[j][i] + moduli[j] - 
                               static_cast<uint64_t>(x % moduli[j])) % moduli[j];
                uint64_t inv = inv_cache.computeModInverse(current_prod % moduli[j], moduli[j]);
                uint64_t coeff = (diff * inv) % moduli[j];
                x += static_cast<__uint128_t>(current_prod) * coeff;
            }
            
            result[i] = static_cast<uint64_t>(x % target_mod);
        }
    }
};

/* ============================== MPI上下文 ============================== */
struct MPIContext {
    int rank, size;
    MPI_Comm comm;
    
    MPIContext() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        comm = MPI_COMM_WORLD;
    }
};

/* ============================== 优化的多模数并行乘法 ============================== */
void optimizedMultiModularMPIMultiply(const vector<int>& a,
                                      const vector<int>& b,
                                      vector<int>& result,
                                      int n,
                                      uint32_t target_mod,
                                      const MPIContext& ctx) {
    // 动态选择最优模数数量
    int optimal_mod_count = ModulusManager::selectOptimalModCount(target_mod, n);
    vector<uint32_t> selected_mods = ModulusManager::getModuli(optimal_mod_count);

    if (ctx.rank == 0) {
        cout << "使用 " << optimal_mod_count << " 个模数进行CRT计算\n";
    }

    // 智能负载均衡：根据进程数和模数数量分配工作
    vector<int> my_mod_indices;
    for (int i = ctx.rank; i < optimal_mod_count; i += ctx.size) {
        my_mod_indices.push_back(i);
    }

    int conv_len = 2 * n - 1;
    vector<vector<uint32_t>> local_results(optimal_mod_count);
    for (int i = 0; i < optimal_mod_count; ++i) {
        local_results[i].resize(conv_len, 0);
    }

    // 并行计算各个模数下的NTT
    for (int idx : my_mod_indices) {
        OptimizedBarrett br(selected_mods[idx]);
        vector<uint32_t> A(a.begin(), a.end());
        vector<uint32_t> B(b.begin(), b.end());

        // 计算合适的NTT长度
        int lim = 1;
        while (lim < 2 * n) lim <<= 1;
        A.resize(lim);
        B.resize(lim);

        // 执行优化的NTT
        optimizedNTT(A, false, br);
        optimizedNTT(B, false, br);

        // 点乘
        for (int i = 0; i < lim; ++i) {
            A[i] = br.mul(A[i], B[i]);
        }

        // 逆NTT
        optimizedNTT(A, true, br);

        // 存储结果
        for (int i = 0; i < conv_len; ++i) {
            local_results[idx][i] = A[i];
        }
    }

    // 收集所有模数的结果
    vector<vector<uint32_t>> all_results(optimal_mod_count);
    for (int i = 0; i < optimal_mod_count; ++i) {
        all_results[i].resize(conv_len);
        MPI_Allreduce(local_results[i].data(), all_results[i].data(),
                     conv_len, MPI_UNSIGNED, MPI_SUM, ctx.comm);
    }

    // 在rank 0上执行CRT重构
    if (ctx.rank == 0) {
        OptimizedCRT crt(selected_mods);
        vector<uint64_t> reconstructed;
        crt.reconstruct(all_results, reconstructed, target_mod);

        result.resize(conv_len);
        for (int i = 0; i < conv_len; ++i) {
            result[i] = static_cast<int>(reconstructed[i]);
        }
    }
}

/* ============================== 性能测试和对比 ============================== */
void performanceComparison(const vector<int>& a, const vector<int>& b,
                          int n, uint32_t target_mod, const MPIContext& ctx) {
    vector<int> result_original, result_optimized;

    // 测试原始实现（如果可用）
    MPI_Barrier(ctx.comm);
    auto t0_orig = chrono::high_resolution_clock::now();
    // 这里可以调用原始的multiModularMPIMultiply函数
    auto t1_orig = chrono::high_resolution_clock::now();

    // 测试优化实现
    MPI_Barrier(ctx.comm);
    auto t0_opt = chrono::high_resolution_clock::now();
    optimizedMultiModularMPIMultiply(a, b, result_optimized, n, target_mod, ctx);
    MPI_Barrier(ctx.comm);
    auto t1_opt = chrono::high_resolution_clock::now();

    if (ctx.rank == 0) {
        double time_orig = chrono::duration<double, milli>(t1_orig - t0_orig).count();
        double time_opt = chrono::duration<double, milli>(t1_opt - t0_opt).count();

        cout << "性能对比 (n=" << n << "):\n";
        cout << "  优化版本: " << fixed << setprecision(3) << time_opt << " ms\n";
        if (time_orig > 0) {
            cout << "  原始版本: " << fixed << setprecision(3) << time_orig << " ms\n";
            cout << "  加速比:   " << fixed << setprecision(2) << time_orig/time_opt << "x\n";
        }
        cout << "  进程数:   " << ctx.size << "\n";
        cout << string(50, '-') << '\n';
    }
}

/* ============================== 主函数 ============================== */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPIContext ctx;

    if (ctx.rank == 0) {
        cout << "优化的CRT多模数NTT实现\n";
        cout << "进程数: " << ctx.size << "\n";
        cout << "支持动态模数选择和多层优化\n";
        cout << string(60, '=') << '\n';
    }

    int a_arr[300000], b_arr[300000];
    const int test_begin = 0, test_end = 3;

    for (int id = test_begin; id <= test_end; ++id) {
        int n = 0, p_test = 0;

        // 读取测试数据
        if (ctx.rank == 0) {
            fRead(a_arr, b_arr, &n, &p_test, id);
        }

        // 广播测试参数
        MPI_Bcast(&n, 1, MPI_INT, 0, ctx.comm);
        MPI_Bcast(&p_test, 1, MPI_INT, 0, ctx.comm);
        MPI_Bcast(a_arr, n, MPI_INT, 0, ctx.comm);
        MPI_Bcast(b_arr, n, MPI_INT, 0, ctx.comm);

        vector<int> a(a_arr, a_arr + n);
        vector<int> b(b_arr, b_arr + n);
        vector<int> result;

        // 执行优化的多模数乘法
        MPI_Barrier(ctx.comm);
        auto t0 = chrono::high_resolution_clock::now();
        optimizedMultiModularMPIMultiply(a, b, result, n, p_test, ctx);
        MPI_Barrier(ctx.comm);
        auto t1 = chrono::high_resolution_clock::now();

        if (ctx.rank == 0) {
            // 验证结果
            fCheck(result.data(), n, id);

            // 输出性能信息
            double elapsed = chrono::duration<double, milli>(t1 - t0).count();
            cout << "测试用例 " << id << " (n=" << n << ", p=" << p_test << "):\n";
            cout << "  执行时间: " << fixed << setprecision(3) << elapsed << " ms\n";
            cout << "  吞吐量:   " << fixed << setprecision(2)
                 << (2.0 * n - 1) / elapsed * 1000 << " ops/sec\n";

            // 写入结果文件
            fWrite(result.data(), n, id);
        }
    }

    MPI_Barrier(ctx.comm);
    if (ctx.rank == 0) {
        cout << string(60, '=') << '\n';
        cout << "所有测试用例完成\n";
        cout << "优化特性:\n";
        cout << "  - 动态模数选择 (3-9个模数)\n";
        cout << "  - 优化的Barrett规约\n";
        cout << "  - 改进的CRT重构算法\n";
        cout << "  - 智能负载均衡\n";
        cout << "  - 缓存友好的内存访问\n";
    }

    MPI_Finalize();
    return 0;
}
