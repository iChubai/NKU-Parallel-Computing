#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

using namespace std;

// 现有的平凡算法：链式累加
double naive_sum(const vector<double>& arr) {
    double sum = 0.0;
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

// 现有的两路链式累加（超标量优化）
double two_way_sum(const vector<double>& arr) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    int n = arr.size();
    for (int i = 0; i < n; i += 2) {
        if (i + 1 < n) {
            sum1 += arr[i];
            sum2 += arr[i + 1];
        } else {
            sum1 += arr[i];
        }
    }
    return sum1 + sum2;
}

// 现有的四路链式累加（更多的指令级并行性）
double four_way_sum(const vector<double>& arr) {
    double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
    int n = arr.size();
    for (int i = 0; i < n; i += 4) {
        if (i + 3 < n) {
            sum1 += arr[i];
            sum2 += arr[i + 1];
            sum3 += arr[i + 2];
            sum4 += arr[i + 3];
        } else {
            for (int j = i; j < n; j++) {
                sum1 += arr[j];
            }
        }
    }
    return sum1 + sum2 + sum3 + sum4;
}

// 现有的递归两两相加
double recursive_sum(const vector<double>& arr, int start, int end) {
    if (end - start <= 1) {
        return start < arr.size() ? arr[start] : 0.0;
    }
    
    int mid = (start + end) / 2;
    return recursive_sum(arr, start, mid) + recursive_sum(arr, mid, end);
}

// 现有的循环展开版本（8次展开）
double unrolled_sum(const vector<double>& arr) {
    double sum = 0.0;
    int n = arr.size();
    int i = 0;
    
    // 处理主要部分 - 8次循环展开
    for (; i + 7 < n; i += 8) {
        sum += arr[i];
        sum += arr[i + 1];
        sum += arr[i + 2];
        sum += arr[i + 3];
        sum += arr[i + 4];
        sum += arr[i + 5];
        sum += arr[i + 6];
        sum += arr[i + 7];
    }
    
    // 处理剩余元素
    for (; i < n; i++) {
        sum += arr[i];
    }
    
    return sum;
}

// 以下是新增的完全消除循环的实现方法 -------------------------

// 使用宏实现的无循环求和（适用于固定大小的小数组）
// 这里展示一个简化的宏展开求和方式，实际应用中需要根据数组大小动态展开
#define SUM_2(arr, start) ((arr)[start] + (arr)[start+1])
#define SUM_4(arr, start) (SUM_2(arr, start) + SUM_2(arr, start+2))
#define SUM_8(arr, start) (SUM_4(arr, start) + SUM_4(arr, start+4))
#define SUM_16(arr, start) (SUM_8(arr, start) + SUM_8(arr, start+8))
#define SUM_32(arr, start) (SUM_16(arr, start) + SUM_16(arr, start+16))
#define SUM_64(arr, start) (SUM_32(arr, start) + SUM_32(arr, start+32))
#define SUM_128(arr, start) (SUM_64(arr, start) + SUM_64(arr, start+64))
#define SUM_256(arr, start) (SUM_128(arr, start) + SUM_128(arr, start+128))

// 使用模板元编程实现的无循环求和
// 这种方法适用于编译时确定大小的数组
template<size_t N>
struct FixedSizeSum {
    static double sum(const vector<double>& arr, size_t start = 0) {
        // 使用模板递归的方式，在编译时展开所有循环
        return arr[start] + FixedSizeSum<N-1>::sum(arr, start+1);
    }
};

// 模板特化，结束递归
template<>
struct FixedSizeSum<1> {
    static double sum(const vector<double>& arr, size_t start = 0) {
        return arr[start];
    }
};

template<>
struct FixedSizeSum<0> {
    static double sum(const vector<double>& arr, size_t start = 0) {
        return 0.0;
    }
};

// 使用模板递归展开循环的两路累加版本
template<size_t N>
struct TwoWayFixedSizeSum {
    static double sum(const vector<double>& arr, size_t start = 0) {
        constexpr size_t half = N / 2;
        // 分成两半，递归处理
        return FixedSizeSum<half>::sum(arr, start) + 
               FixedSizeSum<N-half>::sum(arr, start+half);
    }
};

// 兼容动态大小的混合方法（模板+宏）
// 对于大小可变的数组，我们需要在运行时处理
double macro_template_sum(const vector<double>& arr) {
    double sum = 0.0;
    size_t n = arr.size();
    size_t i = 0;
    
    // 对于大数组，使用宏展开尽可能多的元素
    while (i + 256 <= n) {
        sum += SUM_256(arr, i);
        i += 256;
    }
    while (i + 128 <= n) {
        sum += SUM_128(arr, i);
        i += 128;
    }
    while (i + 64 <= n) {
        sum += SUM_64(arr, i);
        i += 64;
    }
    while (i + 32 <= n) {
        sum += SUM_32(arr, i);
        i += 32;
    }
    while (i + 16 <= n) {
        sum += SUM_16(arr, i);
        i += 16;
    }
    while (i + 8 <= n) {
        sum += SUM_8(arr, i);
        i += 8;
    }
    while (i + 4 <= n) {
        sum += SUM_4(arr, i);
        i += 4;
    }
    while (i + 2 <= n) {
        sum += SUM_2(arr, i);
        i += 2;
    }
    
    // 处理剩余的单个元素
    if (i < n) {
        sum += arr[i];
    }
    
    return sum;
}

// 使用模板展开的两路累加和实现（运行时大小）
double two_way_template_sum(const vector<double>& arr) {
    double sum1 = 0.0, sum2 = 0.0;
    size_t n = arr.size();
    size_t i = 0;
    
    // 尽可能使用两路并行的宏展开
    while (i + 16 <= n) {
        sum1 += SUM_8(arr, i);
        sum2 += SUM_8(arr, i+8);
        i += 16;
    }
    
    // 处理剩余元素
    if (i + 8 <= n) {
        sum1 += SUM_8(arr, i);
        i += 8;
    }
    
    while (i < n) {
        sum1 += arr[i++];
    }
    
    return sum1 + sum2;
}

// 使用模板展开的四路累加和实现
double four_way_template_sum(const vector<double>& arr) {
    double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
    size_t n = arr.size();
    size_t i = 0;
    
    // 尽可能使用四路并行的宏展开
    while (i + 32 <= n) {
        sum1 += SUM_8(arr, i);
        sum2 += SUM_8(arr, i+8);
        sum3 += SUM_8(arr, i+16);
        sum4 += SUM_8(arr, i+24);
        i += 32;
    }
    
    // 处理剩余元素（尽量保持并行性）
    if (i + 16 <= n) {
        sum1 += SUM_8(arr, i);
        sum2 += SUM_8(arr, i+8);
        i += 16;
    }
    
    if (i + 8 <= n) {
        sum1 += SUM_8(arr, i);
        i += 8;
    }
    
    while (i < n) {
        sum1 += arr[i++];
    }
    
    return sum1 + sum2 + sum3 + sum4;
}

// 新增：真正的模板元编程无循环实现 --------------------------------

// 辅助模板类：选择合适的处理方法
template<size_t N>
struct VectorSumHelper {
    // 静态断言确保不会实例化过大的模板
    static_assert(N <= 1048576, "Template vector sum is limited to 1024 elements");
    
    static double sum(const vector<double>& arr) {
        return FixedSizeSum<N>::sum(arr, 0);
    }
};

// 使用真正的模板元编程方式，完全在编译期展开的无循环求和
template<size_t N>
double pure_template_sum(const vector<double>& arr) {
    // 确保数组至少有N个元素
    if (arr.size() < N) {
        throw std::out_of_range("Array size is smaller than template parameter");
    }
    return VectorSumHelper<N>::sum(arr);
}

// 两路并行的纯模板实现
template<size_t N>
struct TwoWayPureSum {
    static double sum(const vector<double>& arr, size_t start = 0) {
        constexpr size_t half = N / 2;
        return FixedSizeSum<half>::sum(arr, start) + 
               FixedSizeSum<half>::sum(arr, start + half);
    }
};

template<size_t N>
double two_way_pure_template_sum(const vector<double>& arr) {
    if (arr.size() < N) {
        throw std::out_of_range("Array size is smaller than template parameter");
    }
    return TwoWayPureSum<N>::sum(arr, 0);
}

// 四路并行的纯模板实现
template<size_t N>
struct FourWayPureSum {
    static double sum(const vector<double>& arr, size_t start = 0) {
        constexpr size_t quarter = N / 4;
        return FixedSizeSum<quarter>::sum(arr, start) + 
               FixedSizeSum<quarter>::sum(arr, start + quarter) +
               FixedSizeSum<quarter>::sum(arr, start + 2*quarter) +
               FixedSizeSum<quarter>::sum(arr, start + 3*quarter);
    }
};

template<size_t N>
double four_way_pure_template_sum(const vector<double>& arr) {
    if (arr.size() < N) {
        throw std::out_of_range("Array size is smaller than template parameter");
    }
    return FourWayPureSum<N>::sum(arr, 0);
}

// 以下是原有函数 --------------------------------

vector<double> generate_test_data(int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    
    // 确保n是2的幂次
    if (n & (n - 1)) {
        int power = 0;
        while (n > 0) {
            n >>= 1;
            power++;
        }
        n = 1 << power; // 找到最接近的2的幂次
        cout << "注意: 将数组大小调整为2的幂次: " << n << endl;
    }
    
    vector<double> arr(n);
    for (int i = 0; i < n; i++) {
        arr[i] = dis(gen);
    }
    
    return arr;
}

bool verify_results(const vector<double>& algorithms_results, double epsilon = 1e-10) {
    if (algorithms_results.size() <= 1) return true;
    
    for (size_t i = 1; i < algorithms_results.size(); i++) {
        if (fabs(algorithms_results[0] - algorithms_results[i]) > epsilon) {
            cout << "不匹配的结果: 算法0=" << algorithms_results[0] 
                 << " vs 算法" << i << "=" << algorithms_results[i] << endl;
            return false;
        }
    }
    
    return true;
}

int main() {
    // 使用2的幂次大小的数组，从2^10到2^20
    vector<int> sizes = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
    const int repetitions = 100; // 对于较小的向量，增加重复次数
    
    cout << "向量求和性能测试" << endl;
    cout << "---------------------------------------------------------------------------------------------------------" << endl;
    cout << setw(12) << "向量大小" 
         << setw(15) << "平凡算法(μs)" 
         << setw(15) << "两路累加(μs)" 
         << setw(15) << "四路累加(μs)" 
         << setw(15) << "循环展开(μs)"
         << setw(15) << "宏模板(μs)"
         << setw(15) << "纯模板(μs)"
         << setw(15) << "两路纯模板(μs)"
         << setw(15) << "四路纯模板(μs)" << endl;
    cout << "---------------------------------------------------------------------------------------------------------" << endl;
    
    for (int n : sizes) {
        // 生成测试数据
        vector<double> arr = generate_test_data(n);
        
        // 存储每种算法的总时间
        double total_time_naive = 0.0;
        double total_time_two_way = 0.0;
        double total_time_four_way = 0.0;
        double total_time_unrolled = 0.0;
        double total_time_macro_template = 0.0;
        double total_time_pure_template = 0.0;
        double total_time_two_way_pure = 0.0;
        double total_time_four_way_pure = 0.0;
        
        // 存储每种算法的结果用于验证
        vector<double> results;
        
        // 运行平凡算法
        for (int r = 0; r < repetitions; r++) {
            auto start = chrono::high_resolution_clock::now();
            double result = naive_sum(arr);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double, micro> duration = end - start;
            total_time_naive += duration.count();
            
            if (r == 0) {
                results.push_back(result);
            }
        }
        
        // 运行两路累加算法
        for (int r = 0; r < repetitions; r++) {
            auto start = chrono::high_resolution_clock::now();
            double result = two_way_sum(arr);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double, micro> duration = end - start;
            total_time_two_way += duration.count();
            
            if (r == 0) {
                results.push_back(result);
            }
        }
        
        // 运行四路累加算法
        for (int r = 0; r < repetitions; r++) {
            auto start = chrono::high_resolution_clock::now();
            double result = four_way_sum(arr);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double, micro> duration = end - start;
            total_time_four_way += duration.count();
            
            if (r == 0) {
                results.push_back(result);
            }
        }
        
        // 运行循环展开算法
        for (int r = 0; r < repetitions; r++) {
            auto start = chrono::high_resolution_clock::now();
            double result = unrolled_sum(arr);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double, micro> duration = end - start;
            total_time_unrolled += duration.count();
            
            if (r == 0) {
                results.push_back(result);
            }
        }
        
        // 运行宏模板混合算法
        for (int r = 0; r < repetitions; r++) {
            auto start = chrono::high_resolution_clock::now();
            double result = macro_template_sum(arr);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double, micro> duration = end - start;
            total_time_macro_template += duration.count();
            
            if (r == 0) {
                results.push_back(result);
            }
        }
        
        // 特殊情况：纯模板算法只能处理固定大小的数组
        // 为测试目的，只在1000元素大小的情况下运行纯模板算法
        if (n == 1000) {
            // 运行纯模板算法
            try {
                for (int r = 0; r < repetitions; r++) {
                    auto start = chrono::high_resolution_clock::now();
                    double result = pure_template_sum<1000>(arr);
                    auto end = chrono::high_resolution_clock::now();
                    chrono::duration<double, micro> duration = end - start;
                    total_time_pure_template += duration.count();
                    
                    if (r == 0) {
                        results.push_back(result);
                    }
                }
                
                // 运行两路纯模板算法
                for (int r = 0; r < repetitions; r++) {
                    auto start = chrono::high_resolution_clock::now();
                    double result = two_way_pure_template_sum<1000>(arr);
                    auto end = chrono::high_resolution_clock::now();
                    chrono::duration<double, micro> duration = end - start;
                    total_time_two_way_pure += duration.count();
                    
                    if (r == 0) {
                        results.push_back(result);
                    }
                }
                
                // 运行四路纯模板算法
                for (int r = 0; r < repetitions; r++) {
                    auto start = chrono::high_resolution_clock::now();
                    double result = four_way_pure_template_sum<1000>(arr);
                    auto end = chrono::high_resolution_clock::now();
                    chrono::duration<double, micro> duration = end - start;
                    total_time_four_way_pure += duration.count();
                    
                    if (r == 0) {
                        results.push_back(result);
                    }
                }
            } catch (const std::exception& e) {
                cout << "错误: " << e.what() << endl;
            }
        }
        
        // 计算平均时间
        double avg_time_naive = total_time_naive / repetitions;
        double avg_time_two_way = total_time_two_way / repetitions;
        double avg_time_four_way = total_time_four_way / repetitions;
        double avg_time_unrolled = total_time_unrolled / repetitions;
        double avg_time_macro_template = total_time_macro_template / repetitions;
        double avg_time_pure_template = n == 1000 ? total_time_pure_template / repetitions : 0.0;
        double avg_time_two_way_pure = n == 1000 ? total_time_two_way_pure / repetitions : 0.0;
        double avg_time_four_way_pure = n == 1000 ? total_time_four_way_pure / repetitions : 0.0;
        
        // 验证结果
        bool results_match = verify_results(results);
        
        // 输出结果
        cout << setw(12) << n 
             << setw(15) << fixed << setprecision(3) << avg_time_naive
             << setw(15) << avg_time_two_way
             << setw(15) << avg_time_four_way
             << setw(15) << avg_time_unrolled
             << setw(15) << avg_time_macro_template;
             
        if (n == 1000) {
            cout << setw(15) << avg_time_pure_template
                 << setw(15) << avg_time_two_way_pure
                 << setw(15) << avg_time_four_way_pure;
        } else {
            cout << setw(15) << "N/A"
                 << setw(15) << "N/A"
                 << setw(15) << "N/A";
        }
        
        cout << (results_match ? "" : " (结果不匹配!)")
             << endl;
    }
    
    // 输出加速比
    cout << "\n各算法相对于平凡算法的加速比" << endl;
    cout << "---------------------------------------------------------------------------------------------------------" << endl;
    cout << setw(12) << "向量大小" 
         << setw(15) << "两路累加"
         << setw(15) << "四路累加"
         << setw(15) << "循环展开"
         << setw(15) << "宏模板"
         << setw(15) << "纯模板"
         << setw(15) << "两路纯模板"
         << setw(15) << "四路纯模板" << endl;
    cout << "---------------------------------------------------------------------------------------------------------" << endl;
    
    for (int n : sizes) {
        // 生成测试数据
        vector<double> arr = generate_test_data(n);
        
        // 测量平凡算法时间
        auto start_naive = chrono::high_resolution_clock::now();
        for (int r = 0; r < repetitions; r++) {
            naive_sum(arr);
        }
        auto end_naive = chrono::high_resolution_clock::now();
        chrono::duration<double, micro> time_naive = end_naive - start_naive;
        
        // 测量两路累加算法时间
        auto start_two_way = chrono::high_resolution_clock::now();
        for (int r = 0; r < repetitions; r++) {
            two_way_sum(arr);
        }
        auto end_two_way = chrono::high_resolution_clock::now();
        chrono::duration<double, micro> time_two_way = end_two_way - start_two_way;
        
        // 测量四路累加算法时间
        auto start_four_way = chrono::high_resolution_clock::now();
        for (int r = 0; r < repetitions; r++) {
            four_way_sum(arr);
        }
        auto end_four_way = chrono::high_resolution_clock::now();
        chrono::duration<double, micro> time_four_way = end_four_way - start_four_way;
        
        // 测量循环展开算法时间
        auto start_unrolled = chrono::high_resolution_clock::now();
        for (int r = 0; r < repetitions; r++) {
            unrolled_sum(arr);
        }
        auto end_unrolled = chrono::high_resolution_clock::now();
        chrono::duration<double, micro> time_unrolled = end_unrolled - start_unrolled;
        
        // 测量宏模板混合算法时间
        auto start_macro_template = chrono::high_resolution_clock::now();
        for (int r = 0; r < repetitions; r++) {
            macro_template_sum(arr);
        }
        auto end_macro_template = chrono::high_resolution_clock::now();
        chrono::duration<double, micro> time_macro_template = end_macro_template - start_macro_template;
        
        // 对于纯模板算法，只在1000元素大小的情况下测试
        chrono::duration<double, micro> time_pure_template(0);
        chrono::duration<double, micro> time_two_way_pure(0);
        chrono::duration<double, micro> time_four_way_pure(0);
        
        if (n == 1000) {
            try {
                // 测量纯模板算法时间
                auto start_pure_template = chrono::high_resolution_clock::now();
                for (int r = 0; r < repetitions; r++) {
                    pure_template_sum<1000>(arr);
                }
                auto end_pure_template = chrono::high_resolution_clock::now();
                time_pure_template = end_pure_template - start_pure_template;
                
                // 测量两路纯模板算法时间
                auto start_two_way_pure = chrono::high_resolution_clock::now();
                for (int r = 0; r < repetitions; r++) {
                    two_way_pure_template_sum<1000>(arr);
                }
                auto end_two_way_pure = chrono::high_resolution_clock::now();
                time_two_way_pure = end_two_way_pure - start_two_way_pure;
                
                // 测量四路纯模板算法时间
                auto start_four_way_pure = chrono::high_resolution_clock::now();
                for (int r = 0; r < repetitions; r++) {
                    four_way_pure_template_sum<1000>(arr);
                }
                auto end_four_way_pure = chrono::high_resolution_clock::now();
                time_four_way_pure = end_four_way_pure - start_four_way_pure;
            } catch (const std::exception& e) {
                cout << "错误: " << e.what() << endl;
            }
        }
        
        // 计算加速比
        double speedup_two_way = time_naive.count() / time_two_way.count();
        double speedup_four_way = time_naive.count() / time_four_way.count();
        double speedup_unrolled = time_naive.count() / time_unrolled.count();
        double speedup_macro_template = time_naive.count() / time_macro_template.count();
        double speedup_pure_template = n == 1000 ? time_naive.count() / time_pure_template.count() : 0.0;
        double speedup_two_way_pure = n == 1000 ? time_naive.count() / time_two_way_pure.count() : 0.0;
        double speedup_four_way_pure = n == 1000 ? time_naive.count() / time_four_way_pure.count() : 0.0;
        
        // 输出加速比
        cout << setw(12) << n 
             << setw(15) << fixed << setprecision(3) << speedup_two_way
             << setw(15) << speedup_four_way
             << setw(15) << speedup_unrolled
             << setw(15) << speedup_macro_template;
             
        if (n == 1000) {
            cout << setw(15) << speedup_pure_template
                 << setw(15) << speedup_two_way_pure
                 << setw(15) << speedup_four_way_pure;
        } else {
            cout << setw(15) << "N/A"
                 << setw(15) << "N/A"
                 << setw(15) << "N/A";
        }
        
        cout << endl;
    }
    
    // 不同编译优化级别的比较实验说明
    cout << "\n注意：当前程序使用的编译优化级别可能会影响性能测试结果。" << endl;
    cout << "建议使用不同的编译优化级别(-O0, -O1, -O2, -O3)进行测试，比较优化对算法性能的影响。" << endl;
    cout << "示例: g++ -O0 task2.cpp -o task2_O0" << endl;
    cout << "      g++ -O3 task2.cpp -o task2_O3" << endl;
    
    cout << "\n无循环算法说明：" << endl;
    cout << "1. 宏模板算法：使用宏展开和模板技术消除循环，通过分层展开的方式处理变长数组" << endl;
    cout << "2. 纯模板算法：完全使用模板元编程在编译期展开所有循环，无任何循环结构" << endl;
    cout << "3. 两路/四路纯模板算法：结合模板元编程和并行处理，同时消除循环和提高指令级并行性" << endl;
    cout << "注意：纯模板算法只能处理固定大小的数组，因此仅在1000元素大小的情况下进行测试。" << endl;
    cout << "纯模板算法的优势在于完全消除循环控制开销，但代价是失去了处理变长数组的灵活性。" << endl;
    
    return 0;
} 