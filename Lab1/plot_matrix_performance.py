#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
矩阵乘法性能分析和可视化工具
此脚本分析CSV结果文件并生成多种可视化图表
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置英文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用默认英文字体
plt.rcParams['axes.unicode_minus'] = True            # 用来正常显示负号

# 创建输出目录
os.makedirs('results/plots', exist_ok=True)

# 设置图表样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# 定义算法颜色和标记
colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47']
markers = ['o', 's', '^', 'D', 'x', '+']

def analyze_matrix_performance():
    """Analyze matrix multiplication performance and generate charts"""
    print('Analyzing matrix multiplication performance data...')
    
    # 读取CSV结果文件
    csv_file = 'results/matrix_multiplication_results.csv'
    if not os.path.exists(csv_file):
        print(f"Error: Result file {csv_file} not found")
        return
    
    data = pd.read_csv(csv_file)
    print(f"Read {len(data)} test data")
    
    # 1. For each matrix type, plot algorithm performance comparison
    matrix_types = data['MatrixType'].unique()
    
    for matrix_type in matrix_types:
        # 按矩阵类型筛选数据
        type_data = data[data['MatrixType'] == matrix_type]
        
        # 对于每种算法类型（不考虑分块大小）绘制性能比较
        plt.figure(figsize=(12, 8))
        
        # 获取此矩阵类型下的基本算法（不包括不同分块大小的算法）
        base_algs = type_data[type_data['BlockSize'] == 0]['Algorithm'].unique()
        
        for i, alg in enumerate(base_algs):
            # 提取特定算法的数据
            alg_data = type_data[(type_data['Algorithm'] == alg) & (type_data['BlockSize'] == 0)]
            
            plt.plot(alg_data['Size'], alg_data['ExecutionTime(ms)'], 
                    color=colors[i % len(colors)], marker=markers[i % len(markers)],
                    label=alg)
        
        plt.title(f'Performance Comparison of Algorithms for {matrix_type}')
        plt.xlabel('Matrix Size (n*n)')
        plt.ylabel('Execution Time (ms)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图表
        filename = f'results/plots/{matrix_type}_algorithms_comparison.png'
        plt.savefig(filename)
        plt.savefig(filename.replace('.png', '.pdf'))
        plt.close()
    
    # 2. Compare block algorithms at different block sizes
    for matrix_type in matrix_types:
        # 选择特定大小的矩阵（如3000x3000）
        target_size = 3000
        type_data = data[(data['MatrixType'] == matrix_type) & (data['Size'] == target_size)]
        
        # 不同分块大小的分块算法
        block_data = type_data[type_data['Algorithm'] == 'blocked']
        block_opt_data = type_data[type_data['Algorithm'] == 'blocked_optimized']
        
        if len(block_data) > 0 and len(block_opt_data) > 0:
            plt.figure(figsize=(10, 6))
            
            plt.plot(block_data['BlockSize'], block_data['ExecutionTime(ms)'], 
                     color=colors[0], marker=markers[0], label='基本分块')
            plt.plot(block_opt_data['BlockSize'], block_opt_data['ExecutionTime(ms)'], 
                     color=colors[1], marker=markers[1], label='优化分块')
            
            plt.title(f'Performance of {matrix_type} at Different Block Sizes (Matrix Size={target_size})')
            plt.xlabel('Block Size')
            plt.ylabel('Execution Time (ms)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            # 保存图表
            filename = f'results/plots/{matrix_type}_block_size_comparison.png'
            plt.savefig(filename)
            plt.savefig(filename.replace('.png', '.pdf'))
            plt.close()
    
    # 3. Create heatmaps for each algorithm comparing performance across different matrix types and sizes
    algorithms = data['Algorithm'].unique()
    
    for alg in algorithms:
        # 如果是分块算法，只选择块大小为64的结果
        if alg in ['blocked', 'blocked_optimized']:
            alg_data = data[(data['Algorithm'] == alg) & (data['BlockSize'] == 64)]
        else:
            alg_data = data[(data['Algorithm'] == alg) & (data['BlockSize'] == 0)]
        
        if len(alg_data) > 0:
            # 创建数据透视表
            pivot_data = alg_data.pivot_table(
                values='ExecutionTime(ms)', 
                index='MatrixType', 
                columns='Size'
            )
            
            plt.figure(figsize=(12, 8))
            ax = sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlGnBu')
            
            plt.title(f'Performance Heatmap of {alg} Across Different Matrix Types and Sizes')
            plt.xlabel('Matrix Size')
            plt.ylabel('Matrix Type')
            plt.tight_layout()
            
            # 保存图表
            filename = f'results/plots/{alg}_heatmap.png'
            plt.savefig(filename)
            plt.savefig(filename.replace('.png', '.pdf'))
            plt.close()
    
    # 4. Compare all algorithms for a specific matrix type and size
    for matrix_type in matrix_types:
        # 选择特定大小的矩阵（3000x3000）
        target_size = 3000
        type_data = data[(data['MatrixType'] == matrix_type) & (data['Size'] == target_size)]
        
        # 对于分块算法，选择最佳分块大小
        algorithms = []
        execution_times = []
        
        # 收集基本算法（非分块）
        for alg in type_data[type_data['BlockSize'] == 0]['Algorithm'].unique():
            alg_data = type_data[(type_data['Algorithm'] == alg) & (type_data['BlockSize'] == 0)]
            algorithms.append(alg)
            execution_times.append(alg_data['ExecutionTime(ms)'].values[0])
        
        # 收集最佳分块大小的分块算法
        for alg in ['blocked', 'blocked_optimized']:
            if alg in type_data['Algorithm'].unique():
                # 找出性能最好的分块大小
                alg_data = type_data[type_data['Algorithm'] == alg]
                if len(alg_data) > 0:
                    best_block = alg_data.loc[alg_data['ExecutionTime(ms)'].idxmin()]
                    block_size = best_block['BlockSize']
                    algorithms.append(f"{alg} (块大小={block_size})")
                    execution_times.append(best_block['ExecutionTime(ms)'])
        
        # 创建条形图
        plt.figure(figsize=(12, 8))
        
        # 按执行时间排序
        sorted_indices = np.argsort(execution_times)
        sorted_algs = [algorithms[i] for i in sorted_indices]
        sorted_times = [execution_times[i] for i in sorted_indices]
        
        # 计算加速比（相对于朴素算法）
        naive_time = type_data[type_data['Algorithm'] == 'naive']['ExecutionTime(ms)'].values[0]
        speedups = [naive_time / time for time in sorted_times]
        
        # 绘制条形图
        bars = plt.barh(range(len(sorted_algs)), sorted_times, color=[colors[i % len(colors)] for i in range(len(sorted_algs))])
        
        # 添加算法标签
        plt.yticks(range(len(sorted_algs)), sorted_algs)
        plt.xlabel('Execution Time (ms)')
        plt.title(f'Performance Comparison of Algorithms for {matrix_type} (Matrix Size={target_size})')
        
        # 添加加速比标注
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                     f'Speedup: {speedup:.2f}x', va='center')
        
        plt.tight_layout()
        
        # 保存图表
        filename = f'results/plots/{matrix_type}_size{target_size}_performance.png'
        plt.savefig(filename)
        plt.savefig(filename.replace('.png', '.pdf'))
        plt.close()
    
    # 5. Create speedup heatmap
    # Select random matrix type, calculate speedup of all algorithms relative to naive algorithm
    random_data = data[data['MatrixType'] == '随机矩阵'].copy()
    
    # Create a new DataFrame to store speedup
    speedup_data = []
    
    for size in random_data['Size'].unique():
        size_data = random_data[random_data['Size'] == size]
        
        # Get naive algorithm time as baseline
        naive_time = size_data[(size_data['Algorithm'] == 'naive') & (size_data['BlockSize'] == 0)]['ExecutionTime(ms)'].values[0]
        
        # Calculate speedup of other algorithms
        for _, row in size_data.iterrows():
            if row['Algorithm'] != 'naive' or row['BlockSize'] != 0:
                speedup = naive_time / row['ExecutionTime(ms)']
                
                if row['BlockSize'] == 0:
                    alg_name = row['Algorithm']
                else:
                    alg_name = f"{row['Algorithm']} (块大小={row['BlockSize']})"
                
                speedup_data.append({
                    'Size': row['Size'],
                    'Algorithm': alg_name,
                    'Speedup': speedup
                })
    
    speedup_df = pd.DataFrame(speedup_data)
    
    # Create pivot table
    if len(speedup_df) > 0:
        pivot_speedup = speedup_df.pivot_table(
            values='Speedup', 
            index='Algorithm', 
            columns='Size'
        )
        
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(pivot_speedup, annot=True, fmt='.2f', cmap='YlOrRd')
        
        plt.title('Speedup Heatmap of All Algorithms Relative to Naive Algorithm')
        plt.xlabel('Matrix Size')
        plt.ylabel('Algorithm')
        plt.tight_layout()
        
        # Save chart
        filename = 'results/plots/speedup_heatmap.png'
        plt.savefig(filename)
        plt.savefig(filename.replace('.png', '.pdf'))
        plt.close()
    
    print('Matrix multiplication performance analysis completed, charts saved to results/plots directory')

if __name__ == "__main__":
    analyze_matrix_performance() 