#!/bin/bash

echo "========================================================"
echo "        并行计算实验测试与分析自动化脚本"
echo "========================================================"

# 运行Task1（矩阵乘法）扩展测试
echo "运行矩阵乘法（Task1）扩展测试..."
bash ./run_matrix_test.sh

# 收集Task2（向量求和）数据
echo "收集向量求和（Task2）数据..."
bash './collect_data.sh'

# 生成所有图表
echo "生成结果图表..."
python ./plot_results.py

echo "========================================================"
echo "                测试与分析完成！"
echo "========================================================"
echo " - 矩阵乘法结果：results/matrix_multiplication_results.csv"
echo " - 详细分析图表：results/plots/"
echo " - 综合图表：results/"
echo "========================================================"
