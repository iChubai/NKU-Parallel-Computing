#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

using namespace std;

vector<double> naive_algorithm(const vector<vector<double>>& matrix, const vector<double>& vec) {
    int n = matrix.size();
    vector<double> sum(n, 0.0);
    
    for(int i = 0; i < n; i++) {
        sum[i] = 0.0;
        for(int j = 0; j < n; j++) {
            sum[i] += matrix[j][i] * vec[j];
        }
    }
    
    return sum;
}

vector<double> optimized_algorithm(const vector<vector<double>>& matrix, const vector<double>& vec) {
    int n = matrix.size();
    vector<double> sum(n, 0.0);
    
    for(int i = 0; i < n; i++) {
        sum[i] = 0.0;
    }
    
    for(int j = 0; j < n; j++) {
        for(int i = 0; i < n; i++) {
            sum[i] += matrix[j][i] * vec[j];
        }
    }
    
    return sum;
}

void generate_test_data(int n, vector<vector<double>>& matrix, vector<double>& vec) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    
    matrix.resize(n, vector<double>(n));
    vec.resize(n);
    
    for(int i = 0; i < n; i++) {
        vec[i] = dis(gen);
        for(int j = 0; j < n; j++) {
            matrix[i][j] = dis(gen);
        }
    }
}

bool verify_results(const vector<double>& res1, const vector<double>& res2, double epsilon = 1e-10) {
    if(res1.size() != res2.size()) return false;
    
    for(size_t i = 0; i < res1.size(); i++) {
        if(fabs(res1[i] - res2[i]) > epsilon) {
            cout << "不匹配的结果在位置 " << i << ": " << res1[i] << " vs " << res2[i] << endl;
            return false;
        }
    }
    
    return true;
}

int main() {
    vector<int> sizes = {100, 500, 1000, 2000, 3000};
    
    cout << "矩阵向量乘法性能测试" << endl;
    cout << "----------------------------------------" << endl;
    cout << setw(10) << "矩阵大小" << setw(15) << "平凡算法(ms)" << setw(15) << "优化算法(ms)" 
         << setw(15) << "加速比" << endl;
    cout << "----------------------------------------" << endl;
    
    for(int n : sizes) {
        vector<vector<double>> matrix;
        vector<double> vec;
        generate_test_data(n, matrix, vec);
        
        auto start1 = chrono::high_resolution_clock::now();
        vector<double> result1 = naive_algorithm(matrix, vec);
        auto end1 = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration1 = end1 - start1;
        
        auto start2 = chrono::high_resolution_clock::now();
        vector<double> result2 = optimized_algorithm(matrix, vec);
        auto end2 = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration2 = end2 - start2;
        
        bool results_match = verify_results(result1, result2);
        
        double speedup = duration1.count() / duration2.count();
        
        cout << setw(10) << n 
             << setw(15) << fixed << setprecision(3) << duration1.count()
             << setw(15) << duration2.count()
             << setw(15) << speedup
             << (results_match ? "" : " (结果不匹配!)")
             << endl;
    }
    
    cout << "\n\n不同问题规模的性能测试（多次测试平均）" << endl;
    cout << "----------------------------------------" << endl;
    
    const int repetitions = 10; // 重复次数
    vector<int> detailed_sizes = {128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072};
    
    cout << setw(10) << "矩阵大小" << setw(15) << "平凡算法(ms)" << setw(15) << "优化算法(ms)" 
         << setw(15) << "加速比" << endl;
    cout << "----------------------------------------" << endl;
    
    for(int n : detailed_sizes) {
        // 生成测试数据
        vector<vector<double>> matrix;
        vector<double> vec;
        generate_test_data(n, matrix, vec);
        
        double total_time1 = 0.0;
        double total_time2 = 0.0;
        
        for(int r = 0; r < repetitions; r++) {
            // 运行平凡算法并计时
            auto start1 = chrono::high_resolution_clock::now();
            vector<double> result1 = naive_algorithm(matrix, vec);
            auto end1 = chrono::high_resolution_clock::now();
            chrono::duration<double, milli> duration1 = end1 - start1;
            total_time1 += duration1.count();
            
            // 运行优化算法并计时
            auto start2 = chrono::high_resolution_clock::now();
            vector<double> result2 = optimized_algorithm(matrix, vec);
            auto end2 = chrono::high_resolution_clock::now();
            chrono::duration<double, milli> duration2 = end2 - start2;
            total_time2 += duration2.count();
        }
        
        double avg_time1 = total_time1 / repetitions;
        double avg_time2 = total_time2 / repetitions;
        double speedup = avg_time1 / avg_time2;
        
        // 输出结果
        cout << setw(10) << n 
             << setw(15) << fixed << setprecision(3) << avg_time1
             << setw(15) << avg_time2
             << setw(15) << speedup
             << endl;
    }
    
    return 0;
} 