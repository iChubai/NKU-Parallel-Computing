#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel Computing Experiment Visualization
Used to generate various charts for experimental reports
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 确保matplotlib可以显示中文
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("Warning: Failed to set Chinese font, using default English font")
    # 使用默认英文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# 创建results目录（如果不存在）
if not os.path.exists('results'):
    os.makedirs('results')

# 定义全局变量
# 优化级别列表
opt_levels = ['O0', 'O1', 'O2', 'O3']

# 颜色和标记列表，用于区分不同的数据系列
colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47', '#7030A0', '#C00000']
markers = ['o', 's', '^', 'D', 'x', '+', '*', 'v']

# 算法名称列表 - 英文
alg_names = ['Naive', 'Loop Unrolling', 'Blocking', 'Strassen Algorithm', 'Special Optimization', 'Template Meta', 'Pure Template']
speedup_alg_names = ['Loop Unrolling', 'Blocking', 'Strassen Algorithm', 'Special Optimization', 'Template Meta', 'Pure Template']

# 加载task1和task2的绘图函数
from plot_task1 import plot_task1, task1_colors, task1_markers, task1_alg_names, task1_speedup_alg_names
from plot_task2 import plot_task2, task2_colors, task2_markers, task2_alg_names, task2_speedup_alg_names

# 导入矩阵乘法性能分析模块
from plot_matrix_performance import analyze_matrix_performance

def plot_overview():
    """Generate overview charts comparing performance of both tasks"""
    print('Generating overview charts...')
    
    plt.figure(figsize=(12, 8))
    
    # 矩阵乘法的最佳加速比
    max_speedup_t1 = {}
    for opt in opt_levels:
        speedup_filename = f'results/task1_speedup_{opt}.csv'
        if os.path.exists(speedup_filename):
            # 读取CSV文件，注意没有列名
            data = pd.read_csv(speedup_filename, header=None)
            
            # 找出每个矩阵大小下的最佳加速比
            best_speedup = data.iloc[:, 2:].max(axis=1)
            max_speedup_t1[opt] = (data.iloc[:, 1], best_speedup)
    
    # 矩阵乘法扩展版的加速比（如果有结果文件）
    matrix_results_file = 'results/matrix_multiplication_results.csv'
    if os.path.exists(matrix_results_file):
        matrix_data = pd.read_csv(matrix_results_file)
        
        # 为随机矩阵类型创建一个额外的图表
        if '随机矩阵' in matrix_data['MatrixType'].unique():
            random_data = matrix_data[matrix_data['MatrixType'] == '随机矩阵']
            
            # 计算相对于朴素算法的加速比
            sizes = []
            speedups = []
            best_algs = []
            
            for size in random_data['Size'].unique():
                size_data = random_data[random_data['Size'] == size]
                
                # 获取朴素算法的时间
                naive_time = size_data[(size_data['Algorithm'] == 'naive') & (size_data['BlockSize'] == 0)]['ExecutionTime(ms)'].values[0]
                
                # 找出最佳算法（排除朴素算法）
                non_naive_data = size_data[(size_data['Algorithm'] != 'naive') | (size_data['BlockSize'] != 0)]
                if len(non_naive_data) > 0:
                    best_idx = non_naive_data['ExecutionTime(ms)'].idxmin()
                    best_row = non_naive_data.loc[best_idx]
                    
                    # 计算加速比
                    speedup = naive_time / best_row['ExecutionTime(ms)']
                    
                    # 记录结果
                    sizes.append(size)
                    speedups.append(speedup)
                    
                    # 记录最佳算法名
                    if best_row['BlockSize'] == 0:
                        best_algs.append(best_row['Algorithm'])
                    else:
                        best_algs.append(f"{best_row['Algorithm']} (块大小={best_row['BlockSize']})")
            
            # 将结果添加到max_speedup_t1字典
            if sizes and speedups:
                max_speedup_t1['Extended'] = (np.array(sizes), np.array(speedups))
    
    # 向量求和的最佳加速比
    max_speedup_t2 = {}
    for opt in opt_levels:
        speedup_filename = f'results/task2_speedup_{opt}.csv'
        if os.path.exists(speedup_filename):
            # 读取CSV文件，注意没有列名
            data = pd.read_csv(speedup_filename, header=None)
            
            # 找出每个向量大小下的最佳加速比
            best_speedup = data.iloc[:, 2:].max(axis=1)
            max_speedup_t2[opt] = (data.iloc[:, 1], best_speedup)
    
    # 创建两个子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绘制矩阵乘法最佳加速比
    for i, opt in enumerate(opt_levels):
        if opt in max_speedup_t1:
            sizes, speedups = max_speedup_t1[opt]
            axes[0].plot(sizes, speedups, color=colors[i], marker=markers[i], label=opt)
    
    # 绘制扩展版矩阵乘法结果（如果有）
    if 'Extended' in max_speedup_t1:
        sizes, speedups = max_speedup_t1['Extended']
        axes[0].plot(sizes, speedups, color='red', marker='*', linewidth=3, label='扩展版')
    
    axes[0].set_title('Matrix Multiplication Best Speedup')
    axes[0].set_xlabel('Matrix Size (n*n)')
    axes[0].set_ylabel('Best Speedup')
    axes[0].set_xscale('log', base=2)
    axes[0].grid(True)
    axes[0].legend(loc='best')
    
    # 绘制向量求和最佳加速比
    for i, opt in enumerate(opt_levels):
        if opt in max_speedup_t2:
            sizes, speedups = max_speedup_t2[opt]
            axes[1].plot(sizes, speedups, color=colors[i], marker=markers[i], label=opt)
    
    axes[1].set_title('Vector Sum Best Speedup')
    axes[1].set_xlabel('Vector Size')
    axes[1].set_ylabel('Best Speedup')
    axes[1].set_xscale('log', base=2)
    axes[1].grid(True)
    axes[1].legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('results/overview_best_speedup.png')
    plt.savefig('results/overview_best_speedup.pdf')
    plt.close()
    
    # 创建算法性能比较热力图（包括新的矩阵乘法算法）
    plt.figure(figsize=(14, 8))
    
    # 获取矩阵乘法扩展版的算法性能数据（如果有）
    if os.path.exists(matrix_results_file):
        matrix_data = pd.read_csv(matrix_results_file)
        
        if '随机矩阵' in matrix_data['MatrixType'].unique():
            # 选择随机矩阵类型，矩阵大小为3000
            size_data = matrix_data[(matrix_data['MatrixType'] == '随机矩阵') & (matrix_data['Size'] == 3000)]
            
            # 获取朴素算法的时间
            naive_time = size_data[(size_data['Algorithm'] == 'naive') & (size_data['BlockSize'] == 0)]['ExecutionTime(ms)'].values[0]
            
            # 收集所有算法的性能数据
            alg_names = []
            speedups = []
            
            # 收集基本算法（非分块）
            for _, row in size_data[size_data['BlockSize'] == 0].iterrows():
                if row['Algorithm'] != 'naive':  # 排除朴素算法
                    alg_name = f"矩阵乘法 - {row['Algorithm']}"
                    speedup = naive_time / row['ExecutionTime(ms)']
                    
                    alg_names.append(alg_name)
                    speedups.append(speedup)
            
            # 收集最佳分块大小的分块算法
            for alg in ['blocked', 'blocked_optimized']:
                alg_data = size_data[size_data['Algorithm'] == alg]
                if len(alg_data) > 0:
                    best_idx = alg_data['ExecutionTime(ms)'].idxmin()
                    best_row = alg_data.loc[best_idx]
                    
                    alg_name = f"矩阵乘法 - {best_row['Algorithm']} (块大小={best_row['BlockSize']})"
                    speedup = naive_time / best_row['ExecutionTime(ms)']
                    
                    alg_names.append(alg_name)
                    speedups.append(speedup)
            
            # 找到最常见的向量大小
            common_size_t2 = 1024  # 向量大小
            
            # 收集向量求和在O3优化级别下的数据
            opt = 'O3'
            task2_filename = f'results/task2_speedup_{opt}.csv'
            if os.path.exists(task2_filename):
                data = pd.read_csv(task2_filename, header=None)
                
                rows_with_size = data[data.iloc[:, 1] == common_size_t2]
                if not rows_with_size.empty:
                    # 选择有效的算法列
                    task2_perf = rows_with_size.iloc[0, 2:min(7, len(data.columns))]
                    
                    for i, perf in enumerate(task2_perf):
                        alg_name = f"向量求和 - {task2_speedup_alg_names[i]}"
                        speedups.append(perf)
                        alg_names.append(alg_name)
            
            # 准备数据并排序
            if alg_names and speedups:
                # 按加速比排序
                sorted_indices = np.argsort(speedups)[::-1]  # 降序
                sorted_algs = [alg_names[i] for i in sorted_indices]
                sorted_speedups = [speedups[i] for i in sorted_indices]
                
                # 创建横向条形图
                bars = plt.barh(range(len(sorted_algs)), sorted_speedups, 
                               color=[colors[i % len(colors)] for i in range(len(sorted_algs))])
                
                # 添加标签
                plt.yticks(range(len(sorted_algs)), sorted_algs)
                plt.xlabel('Speedup')
                plt.title(f'O3 Optimization Level Algorithm Performance Comparison')
                plt.grid(True, axis='x')
                
                # 在每个条形上标出具体数值
                for i, v in enumerate(sorted_speedups):
                    plt.text(v + 0.1, i, f'{v:.2f}x', va='center')
                
                plt.tight_layout()
                plt.savefig('results/overview_algorithm_performance.png')
                plt.savefig('results/overview_algorithm_performance.pdf')
                plt.close()
    
    print('Overview charts generated and saved in results directory.')

def main():
    """Main function, executes all plotting tasks"""
    # 首先绘制概览图表
    plot_overview()
    
    # 绘制任务1(矩阵乘法)的图表
    plot_task1()
    
    # 绘制任务2(向量求和)的图表
    plot_task2()
    
    # 运行矩阵乘法性能分析（如果已经有结果）
    if os.path.exists('results/matrix_multiplication_results.csv'):
        analyze_matrix_performance()
    
    print('All charts generation complete!')

if __name__ == '__main__':
    main() 