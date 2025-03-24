#!/bin/bash

# 创建输出目录
mkdir -p results

# 编译不同优化级别的程序（添加OpenMP支持）
echo "编译不同优化级别的程序..."
g++ -fopenmp -O0 task1.cpp -o task1_O0
g++ -fopenmp -O1 task1.cpp -o task1_O1
g++ -fopenmp -O2 task1.cpp -o task1_O2
g++ -fopenmp -O3 task1.cpp -o task1_O3
g++ -fopenmp -Ofast task1.cpp -o task1_Ofast

g++ -O0 -ftemplate-depth=1025 task2.cpp -o task2_O0
g++ -O1 -ftemplate-depth=1025 task2.cpp -o task2_O1
g++ -O2 -ftemplate-depth=1025 task2.cpp -o task2_O2
g++ -O3 -ftemplate-depth=1025 task2.cpp -o task2_O3
g++ -Ofast -ftemplate-depth=1025 task2.cpp -o task2_Ofast

# 为任务1创建CSV头
echo "size,blocking,unblocking,speedup" > results/task1_O0.csv
echo "size,blocking,unblocking,speedup" > results/task1_O1.csv
echo "size,blocking,unblocking,speedup" > results/task1_O2.csv
echo "size,blocking,unblocking,speedup" > results/task1_O3.csv
echo "size,blocking,unblocking,speedup" > results/task1_Ofast.csv

# 为任务2创建CSV头
echo "size,naive,two_way,four_way,unrolled,macro_template,pure_template,two_way_pure,four_way_pure" > results/task2_time_O0.csv
echo "size,two_way,four_way,unrolled,macro_template,pure_template,two_way_pure,four_way_pure" > results/task2_speedup_O0.csv

echo "size,naive,two_way,four_way,unrolled,macro_template,pure_template,two_way_pure,four_way_pure" > results/task2_time_O1.csv
echo "size,two_way,four_way,unrolled,macro_template,pure_template,two_way_pure,four_way_pure" > results/task2_speedup_O1.csv

echo "size,naive,two_way,four_way,unrolled,macro_template,pure_template,two_way_pure,four_way_pure" > results/task2_time_O2.csv
echo "size,two_way,four_way,unrolled,macro_template,pure_template,two_way_pure,four_way_pure" > results/task2_speedup_O2.csv

echo "size,naive,two_way,four_way,unrolled,macro_template,pure_template,two_way_pure,four_way_pure" > results/task2_time_O3.csv
echo "size,two_way,four_way,unrolled,macro_template,pure_template,two_way_pure,four_way_pure" > results/task2_speedup_O3.csv

echo "size,naive,two_way,four_way,unrolled,macro_template,pure_template,two_way_pure,four_way_pure" > results/task2_time_Ofast.csv
echo "size,two_way,four_way,unrolled,macro_template,pure_template,two_way_pure,four_way_pure" > results/task2_speedup_Ofast.csv

# 收集Task1数据（扩展版）
echo "收集Task1数据（扩展版）..."
for opt in O3; do
    echo "运行 Task1 - $opt 优化..."
    ./task1_$opt
done

# 收集Task2数据
echo "收集Task2数据..."

collect_task2_data() {
    optimization=$1
    
    # 时间数据部分
    ./task2_$optimization | grep -vE "^注意|^各算法|^无循环|^建议|^示例|^-+$|^\$" | 
    awk 'BEGIN {FS="[ ]+"} 
    /^[ ]+[0-9]+/ && NR < 20 {printf "%d,%f,%f,%f,%f,%f,%f,%f,%f\n", $1, $2, $3, $4, $5, $6, $7, $8, $9}' > results/task2_time_$optimization.csv
    
    # 加速比部分
    ./task2_$optimization | grep -vE "^注意|^向量求和|^无循环|^建议|^示例|^-+$|^\$" | 
    awk 'BEGIN {FS="[ ]+"} 
    /^[ ]+[0-9]+/ && NR > 20 {printf "%d,%f,%f,%f,%f,%f,%f,%f\n", $1, $2, $3, $4, $5, $6, $7, $8}' > results/task2_speedup_$optimization.csv
}

for opt in O0 O1 O2 O3 Ofast; do
    echo "运行 Task2 - $opt 优化..."
    collect_task2_data $opt
done

# 运行矩阵性能分析脚本
echo "运行矩阵性能分析..."
python3 plot_matrix_performance.py

echo "数据收集完成。结果保存在 results/ 目录中。" 