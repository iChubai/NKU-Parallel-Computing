#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <string>
#include <omp.h>

using namespace std;

// 矩阵和向量的类型定义
using Matrix = vector<vector<double>>;
using Vector = vector<double>;

// 1. 朴素算法
Vector naive_algorithm(const Matrix& matrix, const Vector& vec) {
    size_t n = matrix.size();
    Vector sum(n, 0.0);
    
    for(size_t i = 0; i < n; i++) {
        sum[i] = 0.0;
        for(size_t j = 0; j < n; j++) {
            sum[i] += matrix[j][i] * vec[j];
        }
    }
    
    return sum;
}

// 2. 缓存优化算法（原优化算法）
Vector optimized_algorithm(const Matrix& matrix, const Vector& vec) {
    size_t n = matrix.size();
    Vector sum(n, 0.0);
    
    for(size_t i = 0; i < n; i++) {
        sum[i] = 0.0;
    }
    
    for(size_t j = 0; j < n; j++) {
        for(size_t i = 0; i < n; i++) {
            sum[i] += matrix[j][i] * vec[j];
        }
    }
    
    return sum;
}

// 3. OpenMP简单并行算法
Vector parallel_algorithm(const Matrix& A, const Vector& v) {
    size_t n = A.size();
    Vector result(n, 0.0);
    
    #pragma omp parallel for
    for(size_t j = 0; j < n; j++ )
        for(size_t i = 0; i < n; i++)
            result[i] += A[j][i] * v[j];
            
    return result;
}

// 4. OpenMP优化并行算法（带局部求和）
Vector parallel_algorithm_optimized(const Matrix& A, const Vector& v) {
    size_t n = A.size();
    Vector result(n, 0.0);
    
    #pragma omp parallel
    {
        Vector local_result(n, 0.0);
        
        #pragma omp for
        for(size_t j = 0; j < n; j++)
            for(size_t i = 0; i < n; i++)
                local_result[i] += A[j][i] * v[j];
        
        #pragma omp critical
        for(size_t i = 0; i < n; i++)
            result[i] += local_result[i];
    }
    
    return result;
}

// 5. 分块算法
Vector blocked_algorithm(const Matrix& A, const Vector& v, size_t blockSize = 64) {
    size_t n = A.size();
    Vector result(n, 0.0);

    #pragma omp parallel for
    for(size_t j = 0; j < n; j += blockSize)
        for(size_t i = 0; i < n; i += blockSize)
            for(size_t jj = j; jj < min(j+blockSize, n); jj++)
                for(size_t ii = i; ii < min(i+blockSize, n); ii++)
                    result[ii] += A[jj][ii] * v[jj];
    
    return result;
}

// 6. 优化的分块算法
Vector blocked_algorithm_optimized(const Matrix& A, const Vector& v, size_t blockSize = 64) {
    size_t n = A.size();
    Vector result(n, 0.0);
    
    #pragma omp parallel
    {
        Vector local_result(n, 0.0);
        
        #pragma omp for schedule(dynamic)
        for(size_t j = 0; j < n; j += blockSize)
            for(size_t i = 0; i < n; i += blockSize) {
                // 预取下一个块的数据（如果编译器支持）
                if(j + blockSize < n)
                    __builtin_prefetch(&A[j + blockSize][0], 0, 3);
                
                // 处理当前块
                size_t j_end = min(j + blockSize, n);
                size_t i_end = min(i + blockSize, n);
                
                for(size_t jj = j; jj < j_end; jj++) {
                    double vj = v[jj]; 
                    for(size_t ii = i; ii < i_end; ii++)
                        local_result[ii] += A[jj][ii] * vj;
                }
            }
        
        #pragma omp critical
        for(size_t i = 0; i < n; i++)
            result[i] += local_result[i];
    }
    
    return result;
}

// 矩阵类型枚举
enum MatrixType {
    RANDOM,    // 随机矩阵
    IDENTITY,  // 单位矩阵
    SPARSE,    // 稀疏矩阵
    HILBERT    // 希尔伯特矩阵
};

// 生成不同类型的测试数据
void generate_test_data(size_t n, Matrix& matrix, Vector& vec, MatrixType type = RANDOM) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    
    matrix.resize(n, Vector(n));
    vec.resize(n);
    
    // 生成向量（所有类型都使用随机向量）
    for(size_t i = 0; i < n; i++) {
        vec[i] = dis(gen);
    }
    
    // 根据不同类型生成矩阵
    switch(type) {
        case RANDOM:
            // 随机矩阵
            for(size_t i = 0; i < n; i++) {
                for(size_t j = 0; j < n; j++) {
                    matrix[i][j] = dis(gen);
                }
            }
            break;
            
        case IDENTITY:
            // 单位矩阵
            for(size_t i = 0; i < n; i++) {
                for(size_t j = 0; j < n; j++) {
                    matrix[i][j] = (i == j) ? 1.0 : 0.0;
                }
            }
            break;
            
        case SPARSE:
            // 稀疏矩阵（约10%的元素非零）
            for(size_t i = 0; i < n; i++) {
                for(size_t j = 0; j < n; j++) {
                    if(dis(gen) < 0.1) {
                        matrix[i][j] = dis(gen);
                    } else {
                        matrix[i][j] = 0.0;
                    }
                }
            }
            break;
            
        case HILBERT:
            // 希尔伯特矩阵：H(i,j) = 1/(i+j+1)
            for(size_t i = 0; i < n; i++) {
                for(size_t j = 0; j < n; j++) {
                    matrix[i][j] = 1.0 / (i + j + 1);
                }
            }
            break;
    }
}

// 验证结果是否一致
bool verify_results(const Vector& res1, const Vector& res2, double epsilon = 1e-10) {
    if(res1.size() != res2.size()) return false;
    
    for(size_t i = 0; i < res1.size(); i++) {
        if(fabs(res1[i] - res2[i]) > epsilon) {
            cout << "不匹配的结果在位置 " << i << ": " << res1[i] << " vs " << res2[i] << endl;
            return false;
        }
    }
    
    return true;
}

// 获取矩阵类型名称
string get_matrix_type_name(MatrixType type) {
    switch(type) {
        case RANDOM: return "随机矩阵";
        case IDENTITY: return "单位矩阵";
        case SPARSE: return "稀疏矩阵";
        case HILBERT: return "希尔伯特矩阵";
        default: return "未知类型";
    }
}

// 解析矩阵类型
MatrixType parse_matrix_type(const string& type_str) {
    if (type_str == "random") return RANDOM;
    if (type_str == "identity") return IDENTITY;
    if (type_str == "sparse") return SPARSE;
    if (type_str == "hilbert") return HILBERT;
    return RANDOM; // 默认为随机矩阵
}

// 运行单一算法进行性能测试（用于perf工具分析）
void run_single_algorithm_test(MatrixType type, size_t size, const string& algorithm) {
    // 生成测试数据
    Matrix matrix;
    Vector vec;
    generate_test_data(size, matrix, vec, type);
    
    if (algorithm == "naive_only") {
        cout << "运行朴素算法 (矩阵大小=" << size << ")..." << endl;
        Vector result = naive_algorithm(matrix, vec);
    } 
    else if (algorithm == "cache_opt_only") {
        cout << "运行缓存优化算法 (矩阵大小=" << size << ")..." << endl;
        Vector result = optimized_algorithm(matrix, vec);
    } 
    else if (algorithm == "openmp_only") {
        cout << "运行OpenMP优化并行算法 (矩阵大小=" << size << ")..." << endl;
        Vector result = parallel_algorithm_optimized(matrix, vec);
    } 
    else if (algorithm == "blocked_only") {
        cout << "运行分块算法 (矩阵大小=" << size << ", 块大小=64)..." << endl;
        Vector result = blocked_algorithm(matrix, vec, 64);
    }
    else if (algorithm == "quick") {
        // 快速运行所有算法各一次，适用于perf分析
        cout << "快速运行所有算法 (矩阵大小=" << size << ")..." << endl;
        Vector result1 = naive_algorithm(matrix, vec);
        Vector result2 = optimized_algorithm(matrix, vec);
        Vector result3 = parallel_algorithm(matrix, vec);
        Vector result4 = parallel_algorithm_optimized(matrix, vec);
        Vector result5 = blocked_algorithm(matrix, vec, 64);
        Vector result6 = blocked_algorithm_optimized(matrix, vec, 64);
    }
}

int main(int argc, char* argv[]) {
    // 检查是否有命令行参数，用于perf工具的特定测试
    if (argc >= 3) {
        string type_str = argv[1];
        size_t size = stoi(argv[2]);
        string mode = (argc >= 4) ? argv[3] : "full";
        
        // 运行特定的性能测试模式
        if (mode != "full") {
            run_single_algorithm_test(parse_matrix_type(type_str), size, mode);
            return 0;
        }
    }
    
    // 如果没有命令行参数或指定full模式，则执行完整测试
    
    // 设置要测试的矩阵大小
    vector<int> sizes = {500, 1000, 2000, 3000, 4000};
    
    // 要测试的矩阵类型
    vector<MatrixType> matrix_types = {RANDOM, IDENTITY, SPARSE, HILBERT};
    
    // 分块大小
    vector<int> block_sizes = {16, 32, 64, 128, 256};
    
    // 重复测试次数
    const int repetitions = 5;
    
    // 创建CSV文件存储结果
    ofstream csv_file("results/matrix_multiplication_results.csv");
    csv_file << "MatrixType,Size,Algorithm,BlockSize,ExecutionTime(ms)" << endl;
    
    cout << "矩阵向量乘法性能测试" << endl;
    cout << "================================================================" << endl;
    
    // 测试不同类型的矩阵
    for(MatrixType type : matrix_types) {
        cout << "\n测试矩阵类型: " << get_matrix_type_name(type) << endl;
        cout << "----------------------------------------------------------------" << endl;
        
        // 测试不同大小的矩阵
        for(int n : sizes) {
            cout << "矩阵大小: " << n << "x" << n << endl;
            
            // 生成测试数据
            Matrix matrix;
            Vector vec;
            generate_test_data(n, matrix, vec, type);
            
            // 用于存储基准结果的变量
            Vector baseline_result;
            
            // 1. 测试朴素算法（基准算法）
            cout << "  朴素算法: ";
            double total_time1 = 0.0;
            
            for(int r = 0; r < repetitions; r++) {
                auto start = chrono::high_resolution_clock::now();
                Vector result = naive_algorithm(matrix, vec);
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double, milli> duration = end - start;
                total_time1 += duration.count();
                
                // 保存第一次的结果作为基准
                if(r == 0) baseline_result = result;
            }
            
            double avg_time1 = total_time1 / repetitions;
            cout << fixed << setprecision(3) << avg_time1 << " ms" << endl;
            csv_file << get_matrix_type_name(type) << "," << n << ",naive,0," << avg_time1 << endl;
            
            // 2. 测试缓存优化算法
            cout << "  缓存优化算法: ";
            double total_time2 = 0.0;
            
            for(int r = 0; r < repetitions; r++) {
                auto start = chrono::high_resolution_clock::now();
                Vector result = optimized_algorithm(matrix, vec);
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double, milli> duration = end - start;
                total_time2 += duration.count();
                
                // 验证结果
                if(r == 0 && !verify_results(baseline_result, result)) {
                    cout << "警告: 缓存优化算法结果不匹配!" << endl;
                }
            }
            
            double avg_time2 = total_time2 / repetitions;
            double speedup2 = avg_time1 / avg_time2;
            cout << avg_time2 << " ms (加速比: " << speedup2 << ")" << endl;
            csv_file << get_matrix_type_name(type) << "," << n << ",cache_optimized,0," << avg_time2 << endl;
            
            // 3. 测试OpenMP简单并行算法
            cout << "  OpenMP简单并行: ";
            double total_time3 = 0.0;
            
            for(int r = 0; r < repetitions; r++) {
                auto start = chrono::high_resolution_clock::now();
                Vector result = parallel_algorithm(matrix, vec);
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double, milli> duration = end - start;
                total_time3 += duration.count();
                
                // 验证结果
                if(r == 0 && !verify_results(baseline_result, result)) {
                    cout << "警告: OpenMP简单并行算法结果不匹配!" << endl;
                }
            }
            
            double avg_time3 = total_time3 / repetitions;
            double speedup3 = avg_time1 / avg_time3;
            cout << avg_time3 << " ms (加速比: " << speedup3 << ")" << endl;
            csv_file << get_matrix_type_name(type) << "," << n << ",openmp_simple,0," << avg_time3 << endl;
            
            // 4. 测试OpenMP优化并行算法
            cout << "  OpenMP优化并行: ";
            double total_time4 = 0.0;
            
            for(int r = 0; r < repetitions; r++) {
                auto start = chrono::high_resolution_clock::now();
                Vector result = parallel_algorithm_optimized(matrix, vec);
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double, milli> duration = end - start;
                total_time4 += duration.count();
                
                // 验证结果
                if(r == 0 && !verify_results(baseline_result, result)) {
                    cout << "警告: OpenMP优化并行算法结果不匹配!" << endl;
                }
            }
            
            double avg_time4 = total_time4 / repetitions;
            double speedup4 = avg_time1 / avg_time4;
            cout << avg_time4 << " ms (加速比: " << speedup4 << ")" << endl;
            csv_file << get_matrix_type_name(type) << "," << n << ",openmp_optimized,0," << avg_time4 << endl;
            
            // 5. 测试不同分块大小的分块算法
            for(int block_size : block_sizes) {
                cout << "  分块算法 (块大小=" << block_size << "): ";
                double total_time5 = 0.0;
                
                for(int r = 0; r < repetitions; r++) {
                    auto start = chrono::high_resolution_clock::now();
                    Vector result = blocked_algorithm(matrix, vec, block_size);
                    auto end = chrono::high_resolution_clock::now();
                    chrono::duration<double, milli> duration = end - start;
                    total_time5 += duration.count();
                    
                    // 验证结果
                    if(r == 0 && !verify_results(baseline_result, result)) {
                        cout << "警告: 分块算法结果不匹配!" << endl;
                    }
                }
                
                double avg_time5 = total_time5 / repetitions;
                double speedup5 = avg_time1 / avg_time5;
                cout << avg_time5 << " ms (加速比: " << speedup5 << ")" << endl;
                csv_file << get_matrix_type_name(type) << "," << n << ",blocked," << block_size << "," << avg_time5 << endl;
            }
            
            // 6. 测试不同分块大小的优化分块算法
            for(int block_size : block_sizes) {
                cout << "  优化分块算法 (块大小=" << block_size << "): ";
                double total_time6 = 0.0;
                
                for(int r = 0; r < repetitions; r++) {
                    auto start = chrono::high_resolution_clock::now();
                    Vector result = blocked_algorithm_optimized(matrix, vec, block_size);
                    auto end = chrono::high_resolution_clock::now();
                    chrono::duration<double, milli> duration = end - start;
                    total_time6 += duration.count();
                    
                    // 验证结果
                    if(r == 0 && !verify_results(baseline_result, result)) {
                        cout << "警告: 优化分块算法结果不匹配!" << endl;
                    }
                }
                
                double avg_time6 = total_time6 / repetitions;
                double speedup6 = avg_time1 / avg_time6;
                cout << avg_time6 << " ms (加速比: " << speedup6 << ")" << endl;
                csv_file << get_matrix_type_name(type) << "," << n << ",blocked_optimized," << block_size << "," << avg_time6 << endl;
            }
            
            cout << endl;
        }
    }
    
    csv_file.close();
    cout << "测试完成，结果已保存到 results/matrix_multiplication_results.csv" << endl;
    
    return 0;
} 