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

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告: 未能设置中文字体，图表中的中文可能无法正确显示")
    # 使用默认英文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

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
    """分析矩阵乘法性能并生成图表"""
    print('分析矩阵乘法性能数据...')
    
    # 读取CSV结果文件
    csv_file = 'results/matrix_multiplication_results.csv'
    if not os.path.exists(csv_file):
        print(f"错误: 找不到结果文件 {csv_file}")
        return
    
    data = pd.read_csv(csv_file)
    print(f"读取到 {len(data)} 条测试数据")
    
    # 1. 为每种矩阵类型绘制算法性能对比图
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
        
        plt.title(f'{matrix_type}的算法性能比较')
        plt.xlabel('矩阵大小 (n*n)')
        plt.ylabel('执行时间 (毫秒)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图表
        filename = f'results/plots/{matrix_type}_algorithms_comparison.png'
        plt.savefig(filename)
        plt.savefig(filename.replace('.png', '.pdf'))
        plt.close()
    
    # 2. 比较分块算法在不同块大小下的性能
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
            
            plt.title(f'{matrix_type} 在不同分块大小下的性能 (矩阵大小={target_size})')
            plt.xlabel('分块大小')
            plt.ylabel('执行时间 (毫秒)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            # 保存图表
            filename = f'results/plots/{matrix_type}_block_size_comparison.png'
            plt.savefig(filename)
            plt.savefig(filename.replace('.png', '.pdf'))
            plt.close()
    
    # 3. 为每种算法创建热力图，比较不同矩阵类型和大小的性能
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
            
            plt.title(f'{alg} 在不同矩阵类型和大小下的性能热力图')
            plt.xlabel('矩阵大小')
            plt.ylabel('矩阵类型')
            plt.tight_layout()
            
            # 保存图表
            filename = f'results/plots/{alg}_heatmap.png'
            plt.savefig(filename)
            plt.savefig(filename.replace('.png', '.pdf'))
            plt.close()
    
    # 4. 比较所有算法在特定矩阵类型和大小下的性能
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
        plt.xlabel('执行时间 (毫秒)')
        plt.title(f'{matrix_type} 矩阵大小={target_size} 的算法性能比较')
        
        # 添加加速比标注
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                     f'加速比: {speedup:.2f}x', va='center')
        
        plt.tight_layout()
        
        # 保存图表
        filename = f'results/plots/{matrix_type}_size{target_size}_performance.png'
        plt.savefig(filename)
        plt.savefig(filename.replace('.png', '.pdf'))
        plt.close()
    
    # 5. 创建加速比热力图
    # 选择随机矩阵类型，计算所有算法相对于朴素算法的加速比
    random_data = data[data['MatrixType'] == '随机矩阵'].copy()
    
    # 创建一个新的DataFrame存储加速比
    speedup_data = []
    
    for size in random_data['Size'].unique():
        size_data = random_data[random_data['Size'] == size]
        
        # 获取朴素算法的时间作为基准
        naive_time = size_data[(size_data['Algorithm'] == 'naive') & (size_data['BlockSize'] == 0)]['ExecutionTime(ms)'].values[0]
        
        # 计算其他算法的加速比
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
    
    # 创建数据透视表
    if len(speedup_df) > 0:
        pivot_speedup = speedup_df.pivot_table(
            values='Speedup', 
            index='Algorithm', 
            columns='Size'
        )
        
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(pivot_speedup, annot=True, fmt='.2f', cmap='YlOrRd')
        
        plt.title('各算法在不同矩阵大小下的加速比热力图 (基准: 朴素算法)')
        plt.xlabel('矩阵大小')
        plt.ylabel('算法')
        plt.tight_layout()
        
        # 保存图表
        filename = 'results/plots/speedup_heatmap.png'
        plt.savefig(filename)
        plt.savefig(filename.replace('.png', '.pdf'))
        plt.close()
    
    print('矩阵乘法性能分析完成，图表已保存至 results/plots 目录')

if __name__ == "__main__":
    analyze_matrix_performance() 