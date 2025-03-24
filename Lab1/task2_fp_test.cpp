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
        
        // 直接对比各个算法之间的差异
        double diff12 = fabs(sum1 - sum2);
        double diff13 = fabs(sum1 - sum3);
        double diff14 = fabs(sum1 - sum4);
        double diff23 = fabs(sum2 - sum3);
        double diff24 = fabs(sum2 - sum4);
        double diff34 = fabs(sum3 - sum4);
        
        // 输出结果
        cout << setw(10) << n 
             << setw(20) << scientific << setprecision(8) << sum1
             << setw(20) << sum2
             << setw(20) << sum3
             << setw(20) << sum4
             << setw(15) << diff12 << setw(15) << diff13 << setw(15) << diff14 
             << setw(15) << diff23 << setw(15) << diff24 << setw(15) << diff34 << endl;
    
    # // 测试2：Kahan补偿求和专用测试
    # cout << endl << "测试2：Kahan补偿求和与常规求和对比（接近值累加）" << endl;
    # cout << "-----------------------------------------------------------" << endl;
    # cout << setw(10) << "向量大小" 
    #      << setw(20) << "常规累加" 
    #      << setw(20) << "Kahan补偿求和" 
    #      << setw(20) << "长双精度求和"
    #      << setw(20) << "理论精确值"
    #      << setw(15) << "常规误差"
    #      << setw(15) << "Kahan误差" << endl;
    # cout << "-----------------------------------------------------------" << endl;
    
    # for (int n : sizes) {
    #     // 生成Kahan测试数据
    #     vector<double> arr = generate_kahan_test_data(n);
        
    #     // 计算不同方法的累加和
    #     double sum1 = forward_sum(arr);
    #     double sum2 = kahan_sum(arr);
    #     double sum3 = long_double_sum(arr);
        
    #     // 计算理论精确值（这里用简单公式，实际中可能需要更复杂的计算）
    #     double exact = 1.0 * n + 1e-8 * (n * (n-1) / 2) % 100;
        
    #     // 计算误差
    #     double error1 = fabs(sum1 - exact);
    #     double error2 = fabs(sum2 - exact);
        
    #     // 输出结果
    #     cout << setw(10) << n 
    #          << setw(20) << scientific << setprecision(8) << sum1
    #          << setw(20) << sum2
    #          << setw(20) << sum3
    #          << setw(20) << exact
    #          << setw(15) << error1
    #          << setw(15) << error2 << endl;
    # }
    
    # cout << "===========================================================" << endl;
    # cout << "结论：" << endl;
    # cout << "1. 浮点数累加顺序对结果有显著影响，特别是当数据包含不同量级的值时" << endl;
    # cout << "2. 成对累加（分治法）和排序后累加通常能提供更稳定和准确的结果" << endl;
    # cout << "3. Kahan补偿求和算法可以有效减少舍入误差，特别是在累加接近值时" << endl;
    # cout << "4. 在追求高精度计算时，应当考虑使用适当的累加顺序或补偿算法" << endl;
    
    return 0;
}
