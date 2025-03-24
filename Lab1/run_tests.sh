# #!/bin/bash

# echo "========================================================"
# echo "       并行计算实验测试脚本 - 不同优化级别比较"
# echo "========================================================"

# # 创建结果目录
# mkdir -p results
# mkdir -p profiling_results

# # 检查perf是否安装
# if ! command -v perf &> /dev/null; then
#     echo "警告：'perf'工具未安装，无法进行性能分析。"
#     echo "请使用 'sudo apt install linux-tools-common linux-tools-generic' 安装。"
#     HAS_PERF=false
# else
#     HAS_PERF=true
# fi

# # 函数：生成2的幂次大小的测试数据
# # 使用时: generate_powers_of_two 10 20 将生成从2^10到2^20的所有2的幂次
# function generate_powers_of_two() {
#     local min_power=$1
#     local max_power=$2
#     local sizes=()
    
#     for ((i=min_power; i<=max_power; i++)); do
#         sizes+=($(echo "2^$i" | bc))
#     done
    
#     echo "${sizes[@]}"
# }

# # # 任务1：矩阵向量乘法测试
# # echo ""
# # echo "任务1：矩阵向量乘法 - 不同编译优化级别"
# # echo "------------------------------------------------------"

# # # 编译不同优化级别的版本
# # echo "正在编译任务1的不同优化级别版本..."
# # g++ -O0 task1.cpp -o task1_O0
# # g++ -O1 task1.cpp -o task1_O1
# # g++ -O2 task1.cpp -o task1_O2
# # g++ -O3 task1.cpp -o task1_O3 -march=native

# # # 运行不同优化级别的版本
# # echo ""
# # echo "任务1：基本性能测试结果（矩阵大小=1024）"
# # echo "------------------------------------------------------"
# # for opt_level in 0 1 2 3; do
# #     echo "运行 -O$opt_level 优化级别："
# #     ./task1_O$opt_level | grep -E "1024|矩阵大小" | head -n 2
# #     echo ""
# # done

# # # Profiling任务1 (仅使用-O3优化)
# # if [ "$HAS_PERF" = true ]; then
# #     echo ""
# #     echo "任务1：性能分析（使用perf工具）"
# #     echo "------------------------------------------------------"
# #     echo "正在收集性能数据..."
    
# #     # CPU周期和分支预测数据
# #     perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses ./task1_O3 > /dev/null 2> profiling_results/task1_O3_stat.txt
    
# #     # 热点函数分析
# #     perf record -g -o profiling_results/task1_O3.data ./task1_O3 > /dev/null 2>&1
# #     perf report -i profiling_results/task1_O3.data > profiling_results/task1_O3_report.txt
    
# #     echo "性能分析结果已保存到 profiling_results 目录"
    
# #     # 输出摘要
# #     echo ""
# #     echo "任务1：性能分析摘要："
# #     grep -A 10 "Performance counter stats" profiling_results/task1_O3_stat.txt
# #     echo ""
# # fi

# # 任务2：向量求和测试
# echo ""
# echo "任务2：向量求和 - 不同编译优化级别"
# echo "------------------------------------------------------"

# # 编译不同优化级别的版本
# echo "正在编译任务2的不同优化级别版本..."
# g++ -O0 task2.cpp -o task2_O0
# g++ -O1 task2.cpp -o task2_O1
# g++ -O2 task2.cpp -o task2_O2
# g++ -O3 task2.cpp -o task2_O3 -march=native

# # 运行不同优化级别的版本
# echo ""
# echo "任务2：基本性能测试结果（向量大小=1024,2048,4096）"
# echo "------------------------------------------------------"
# for opt_level in 0 1 2 3; do
#     echo "运行 -O$opt_level 优化级别："
#     for size in 1024 2048 4096; do
#         ./task2_O$opt_level | grep -E "^[ ]*$size" | head -n 1
#     done
#     echo ""
# done

# # 创建浮点数运算次序测试版本
# echo ""
# echo "任务2：浮点数运算次序测试"
# echo "------------------------------------------------------"
# echo "正在编译启用/禁用浮点数优化的版本..."

# 创建临时测试文件，添加浮点数运算次序测试代码
cat > task2_fp_test.cpp << 'EOF'
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace std;

// 生成具有特定模式的测试数据以显示浮点数舍入差异
vector<double> generate_fp_test_data(int n) {
    vector<double> arr(n);
    
    // 设置一些非常大的值和非常小的值来放大舍入误差
    for (int i = 0; i < n; i++) {
        if (i % 3 == 0) {
            arr[i] = 1e15 + static_cast<double>(i);  // 非常大的值
        } else if (i % 3 == 1) {
            arr[i] = 1e-15 * static_cast<double>(i); // 非常小的值
        } else {
            arr[i] = static_cast<double>(i);         // 普通值
        }
    }
    
    return arr;
}

// 以不同顺序执行累加，观察结果差异
double forward_sum(const vector<double>& arr) {
    double sum = 0.0;
    for (size_t i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    return sum;
}

double backward_sum(const vector<double>& arr) {
    double sum = 0.0;
    for (int i = arr.size() - 1; i >= 0; i--) {
        sum += arr[i];
    }
    return sum;
}

double pairwise_sum(const vector<double>& arr) {
    vector<double> temp = arr;
    int n = arr.size();
    
    while (n > 1) {
        int half = (n + 1) / 2;
        for (int i = 0; i < n/2; i++) {
            temp[i] = temp[i] + temp[i + half];
        }
        n = half;
    }
    
    return temp[0];
}

double sorted_sum(const vector<double>& arr) {
    vector<double> sorted = arr;
    sort(sorted.begin(), sorted.end(), [](double a, double b) {
        return fabs(a) < fabs(b);  // 按绝对值从小到大排序
    });
    
    double sum = 0.0;
    for (double val : sorted) {
        sum += val;
    }
    return sum;
}

int main() {
    vector<int> sizes = {1024, 2048, 4096, 8192, 16384};
    
    cout << "浮点数累加顺序对结果的影响测试" << endl;
    cout << "----------------------------------------------------------" << endl;
    cout << setw(10) << "向量大小" 
         << setw(20) << "正向累加" 
         << setw(20) << "反向累加" 
         << setw(20) << "成对累加" 
         << setw(20) << "排序后累加" 
         << setw(15) << "最大差异" << endl;
    cout << "----------------------------------------------------------" << endl;
    
    for (int n : sizes) {
        // 生成特殊的浮点数测试数据
        vector<double> arr = generate_fp_test_data(n);
        
        // 计算不同顺序的累加和
        double sum1 = forward_sum(arr);
        double sum2 = backward_sum(arr);
        double sum3 = pairwise_sum(arr);
        double sum4 = sorted_sum(arr);
        
        // 计算最大差异
        double max_diff = 0.0;
        max_diff = max(max_diff, fabs(sum1 - sum2));
        max_diff = max(max_diff, fabs(sum1 - sum3));
        max_diff = max(max_diff, fabs(sum1 - sum4));
        max_diff = max(max_diff, fabs(sum2 - sum3));
        max_diff = max(max_diff, fabs(sum2 - sum4));
        max_diff = max(max_diff, fabs(sum3 - sum4));
        
        // 输出结果
        cout << setw(10) << n 
             << setw(20) << scientific << setprecision(8) << sum1
             << setw(20) << sum2
             << setw(20) << sum3
             << setw(20) << sum4
             << setw(15) << max_diff << endl;
    }
    
    cout << "----------------------------------------------------------" << endl;
    cout << "注意：差异反映了浮点数舍入误差在不同累加顺序下的影响。" << endl;
    cout << "      成对累加和排序后累加通常能提供更稳定和准确的结果。" << endl;
    
    return 0;
}
EOF

# 编译浮点数测试版本
g++ -O3 task2_fp_test.cpp -o task2_fp_test
g++ -O3 -ffast-math task2_fp_test.cpp -o task2_fp_test_fastmath

# 运行浮点数顺序测试
echo "标准浮点数模式结果："
./task2_fp_test

echo ""
echo "启用快速浮点数模式结果(-ffast-math)："
./task2_fp_test_fastmath

# # Profiling任务2 (仅使用-O3优化)
# if [ "$HAS_PERF" = true ]; then
#     echo ""
#     echo "任务2：性能分析（使用perf工具）"
#     echo "------------------------------------------------------"
#     echo "正在收集性能数据..."
    
#     # CPU周期和缓存数据
#     perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses ./task2_O3 > /dev/null 2> profiling_results/task2_O3_stat.txt
    
#     # 热点函数分析
#     perf record -g -o profiling_results/task2_O3.data ./task2_O3 > /dev/null 2>&1
#     perf report -i profiling_results/task2_O3.data > profiling_results/task2_O3_report.txt
    
#     # 性能分析：比较不同算法的热点函数
#     # 注释掉有问题的部分
#     # echo "比较不同求和算法的性能特性..."
#     # for algo in "naive_sum" "two_way_sum" "four_way_sum" "unrolled_sum" "macro_template_sum"; do
#     #     echo "分析 $algo 算法..."
#     #     perf record -g --call-graph dwarf -e cycles:u -o profiling_results/task2_${algo}.data -- ./task2_O3 > /dev/null 2>&1
#     #     perf report -i profiling_results/task2_${algo}.data --stdio --symbol-filter=${algo} > profiling_results/task2_${algo}_report.txt
#     # done
    
#     echo "性能分析结果已保存到 profiling_results 目录"
    
#     # 输出摘要
#     echo ""
#     echo "任务2：性能分析摘要："
#     grep -A 10 "Performance counter stats" profiling_results/task2_O3_stat.txt
    
#     # 分析缓存效率
#     echo ""
#     echo "任务2：缓存效率分析："
#     perf stat -e L1-dcache-loads,L1-dcache-load-misses ./task2_O3 > /dev/null 2> profiling_results/task2_O3_cache.txt
#     grep -A 5 "Performance counter stats" profiling_results/task2_O3_cache.txt
    
#     echo ""
# fi

# # 删除临时文件
# rm -f task2_fp_test.cpp

# echo ""
# echo "========================================================"
# echo "                 所有测试已完成！"
# echo "========================================================"
# echo "结果摘要："
# echo "- 性能分析报告已保存到 profiling_results/ 目录"
# echo "- 不同优化级别的比较已显示在控制台输出中"
# echo "- 浮点数累加顺序测试结果已显示"
# echo ""
# echo "建议分析："
# echo "1. 比较不同优化级别对性能的影响"
# echo "2. 观察浮点数运算顺序对结果精度的影响"
# echo "3. 根据profiling结果分析热点函数和缓存行为"
# echo "4. 对比指令级并行方法的效率差异"
# echo "========================================================" 