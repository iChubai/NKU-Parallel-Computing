#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量求和性能实验Python绘图工具
此脚本从收集的CSV文件中读取数据并生成向量求和(task2)的高质量图表
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告: 未能设置中文字体，图表中的中文可能无法正确显示")
    # 使用默认英文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 创建输出目录
os.makedirs('results', exist_ok=True)

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

# 优化级别和图形样式
opt_levels = ['O0', 'O1', 'O2', 'O3', 'Ofast']
colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30']
markers = ['o', 's', 'd', '^', 'v']

# 算法名称定义 - 使用英文名称
alg_names = ['Naive', 'Dual Path', 'Quad Path', 'Loop Unrolling', 'Macro Template', 'Pure Template', 'Dual Pure Temp', 'Quad Pure Temp']
speedup_alg_names = ['Dual Path', 'Quad Path', 'Loop Unrolling', 'Macro Template', 'Pure Template', 'Dual Pure Temp', 'Quad Pure Temp']

# 导出变量供其他模块使用
task2_colors = colors
task2_markers = markers
task2_alg_names = alg_names
task2_speedup_alg_names = speedup_alg_names

def plot_task2():
    """生成向量求和(Task 2)的图表"""
    print('Generating Vector Sum (Task 2) Charts...')
    
    # 图1: 不同优化级别下的执行时间比较
    fig = plt.figure(figsize=(15, 12))
    
    for o, opt in enumerate(opt_levels):
        time_filename = f'results/task2_time_{opt}.csv'
        if os.path.exists(time_filename):
            # 读取CSV文件，注意没有列名，需要自己提供
            time_data = pd.read_csv(time_filename, header=None)
            print(f"Reading file: {time_filename}, shape: {time_data.shape}")
            
            # 为每个优化级别创建一个子图
            ax = fig.add_subplot(2, 3, o+1)
            
            # 使用第1列作为向量大小(索引0是行号，索引1是第一个数据列)，绘制所有算法的执行时间
            for i in range(2, min(9, len(time_data.columns))):
                if i == 7 and opt != 'O0':  # 只在O0级别显示纯模板算法时间
                    continue
                color_index = (i-2) % len(colors)  # 确保颜色索引不会超出范围
                marker_index = (i-2) % len(markers)  # 确保标记索引不会超出范围
                alg_index = min(i-2, len(alg_names)-1)  # 确保算法名称索引不会超出范围
                
                ax.plot(time_data.iloc[:, 1], time_data.iloc[:, i], 
                       color=colors[color_index], 
                       marker=markers[marker_index], 
                       label=alg_names[alg_index])
            
            ax.set_title(f'Optimization Level {opt} Execution Time')
            ax.set_xlabel('Vector Size')
            ax.set_ylabel('Execution Time (μs)')
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.legend(loc='upper left')
            ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/task2_execution_times.png')
    plt.savefig('results/task2_execution_times.pdf')
    plt.close()
    
    # 图2: 不同优化级别下的加速比
    fig = plt.figure(figsize=(15, 12))
    
    for o, opt in enumerate(opt_levels):
        speedup_filename = f'results/task2_speedup_{opt}.csv'
        if os.path.exists(speedup_filename):
            # 读取CSV文件，注意没有列名，需要自己提供
            speedup_data = pd.read_csv(speedup_filename, header=None)
            print(f"Reading file: {speedup_filename}, shape: {speedup_data.shape}")
            
            # 为每个优化级别创建一个子图
            ax = fig.add_subplot(2, 3, o+1)
            
            # 绘制所有算法的加速比
            for i in range(2, min(9, len(speedup_data.columns))):
                if i >= 6 and opt != 'O0':  # 只在O0级别显示纯模板算法加速比
                    continue
                color_index = (i-1) % len(colors)  # 确保颜色索引不会超出范围
                marker_index = (i-1) % len(markers)  # 确保标记索引不会超出范围
                alg_index = min(i-2, len(speedup_alg_names)-1)  # 确保算法名称索引不会超出范围
                
                ax.plot(speedup_data.iloc[:, 1], speedup_data.iloc[:, i], 
                       color=colors[color_index], 
                       marker=markers[marker_index], 
                       label=speedup_alg_names[alg_index])
            
            ax.set_title(f'Optimization Level {opt} Speedup')
            ax.set_xlabel('Vector Size')
            ax.set_ylabel('Speedup Relative to Naive Algorithm')
            ax.set_xscale('log', base=2)
            ax.legend(loc='best')
            ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/task2_speedup.png')
    plt.savefig('results/task2_speedup.pdf')
    plt.close()
    
    # 图3: 不同优化级别下的最佳算法比较
    plt.figure(figsize=(10, 6))
    
    for i, opt in enumerate(opt_levels):
        speedup_filename = f'results/task2_speedup_{opt}.csv'
        if os.path.exists(speedup_filename):
            # 读取CSV文件，注意没有列名
            speedup_data = pd.read_csv(speedup_filename, header=None)
            
            # 找出每个向量大小下的最佳加速比
            best_speedup = speedup_data.iloc[:, 2:].max(axis=1)
            plt.plot(speedup_data.iloc[:, 1], best_speedup, color=colors[i], marker=markers[i], label=opt)
    
    plt.title('Best Algorithm Speedup Across Optimization Levels')
    plt.xlabel('Vector Size')
    plt.ylabel('Best Speedup')
    plt.xscale('log', base=2)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/task2_best_speedup.png')
    plt.savefig('results/task2_best_speedup.pdf')
    plt.close()
    
    # 图4: 比较所有优化级别下的宏模板算法
    plt.figure(figsize=(10, 6))
    
    for i, opt in enumerate(opt_levels):
        speedup_filename = f'results/task2_speedup_{opt}.csv'
        if os.path.exists(speedup_filename):
            # 读取CSV文件，注意没有列名
            speedup_data = pd.read_csv(speedup_filename, header=None)
            # 宏模板算法通常是第5列
            if len(speedup_data.columns) >= 6:
                plt.plot(speedup_data.iloc[:, 1], speedup_data.iloc[:, 5], color=colors[i], marker=markers[i], label=opt)
    
    plt.title('Macro Template Algorithm Speedup Across Optimization Levels')
    plt.xlabel('Vector Size')
    plt.ylabel('Speedup Relative to Naive Algorithm')
    plt.xscale('log', base=2)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/task2_macro_template_speedup.png')
    plt.savefig('results/task2_macro_template_speedup.pdf')
    plt.close()
    
    # 图5: O3优化级别下的执行时间详细比较
    plt.figure(figsize=(10, 6))
    
    opt = 'O3'
    time_filename = f'results/task2_time_{opt}.csv'
    if os.path.exists(time_filename):
        # 读取CSV文件，注意没有列名
        time_data = pd.read_csv(time_filename, header=None)
        
        for i in range(2, min(7, len(time_data.columns))):
            plt.plot(time_data.iloc[:, 1], time_data.iloc[:, i], color=colors[i-2], marker=markers[i-2], 
                    label=alg_names[i-2])
        
        plt.title(f'Optimization Level {opt} Vector Sum Execution Time')
        plt.xlabel('Vector Size')
        plt.ylabel('Execution Time (μs)')
        plt.xscale('log', base=2)
        plt.yscale('log')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/task2_O3_execution_times.png')
        plt.savefig('results/task2_O3_execution_times.pdf')
        plt.close()
    
    # 图6: O3优化级别下的加速比详细比较
    plt.figure(figsize=(10, 6))
    
    opt = 'O3'
    speedup_filename = f'results/task2_speedup_{opt}.csv'
    if os.path.exists(speedup_filename):
        # 读取CSV文件，注意没有列名
        speedup_data = pd.read_csv(speedup_filename, header=None)
        
        for i in range(2, min(7, len(speedup_data.columns))):
            plt.plot(speedup_data.iloc[:, 1], speedup_data.iloc[:, i], color=colors[i-1], marker=markers[i-1], 
                    label=speedup_alg_names[i-2])
        
        plt.title(f'Optimization Level {opt} Vector Sum Speedup')
        plt.xlabel('Vector Size')
        plt.ylabel('Speedup Relative to Naive Algorithm')
        plt.xscale('log', base=2)
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/task2_O3_speedup.png')
        plt.savefig('results/task2_O3_speedup.pdf')
        plt.close()
    
    # 图7: 最佳算法分析
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    opt = 'O3'
    time_filename = f'results/task2_time_{opt}.csv'
    if os.path.exists(time_filename):
        # 读取CSV文件，注意没有列名
        time_data = pd.read_csv(time_filename, header=None)
        time_values = time_data.iloc[:, 2:min(7, len(time_data.columns))].values
        best_indices = np.argmin(time_values, axis=1)
        
        # 为每种最佳算法选择不同颜色
        unique_best = np.unique(best_indices)
        for i, alg_idx in enumerate(unique_best):
            sizes = time_data.iloc[best_indices == alg_idx, 1].values
            axes[0].scatter(sizes, np.ones(len(sizes))*i, s=100, color=colors[alg_idx], label=alg_names[alg_idx])
        
        axes[0].set_title(f'Optimization Level {opt} Best Algorithm Distribution')
        axes[0].set_xlabel('Vector Size')
        axes[0].set_yticks(range(len(unique_best)))
        axes[0].set_yticklabels([alg_names[idx] for idx in unique_best])
        axes[0].set_xscale('log', base=2)
        axes[0].grid(True)
        axes[0].legend(loc='best')
    
    speedup_filename = f'results/task2_speedup_{opt}.csv'
    if os.path.exists(speedup_filename):
        # 读取CSV文件，注意没有列名
        speedup_data = pd.read_csv(speedup_filename, header=None)
        
        # 寻找最大加速比
        max_speedup = speedup_data.iloc[:, 2:min(7, len(speedup_data.columns))].max(axis=1)
        axes[1].plot(speedup_data.iloc[:, 1], max_speedup, 'k-', linewidth=2, label='Maximum Speedup')
        
        # 在图上标注最佳点
        peak_idx = max_speedup.idxmax()
        peak_speedup = max_speedup[peak_idx]
        peak_size = speedup_data.iloc[peak_idx, 1]
        axes[1].scatter([peak_size], [peak_speedup], s=100, c='r', label='Peak Speedup')
        axes[1].text(peak_size, peak_speedup, f' {peak_speedup:.2f}x', va='bottom')
        
        axes[1].set_title(f'Optimization Level {opt} Maximum Speedup')
        axes[1].set_xlabel('Vector Size')
        axes[1].set_ylabel('Speedup Relative to Naive Algorithm')
        axes[1].set_xscale('log', base=2)
        axes[1].grid(True)
        axes[1].legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('results/task2_O3_best_algorithm_analysis.png')
    plt.savefig('results/task2_O3_best_algorithm_analysis.pdf')
    plt.close()
    
    # 图8: 纯模板算法在1024元素时的性能分析
    plt.figure(figsize=(12, 7))
    
    opt = 'O0'  # 纯模板算法通常在O0级别最有优势
    time_filename = f'results/task2_time_{opt}.csv'
    if os.path.exists(time_filename):
        # 读取CSV文件，注意没有列名
        time_data = pd.read_csv(time_filename, header=None)
        small_data = time_data[time_data.iloc[:, 1] == 1024]
        
        if not small_data.empty:
            # 提取所有算法的时间数据
            times = small_data.iloc[0, 2:].values
            
            # 创建条形图
            bars = plt.bar(range(len(times)), times, color=colors[:len(times)])
            
            # 在每个条形上标出具体数值
            for i, v in enumerate(times):
                plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
            
            plt.title(f'Vector Size 1024 Performance Comparison ({opt} Optimization)')
            plt.ylabel('Execution Time (μs)')
            plt.xticks(range(len(alg_names[:len(times)])), alg_names[:len(times)], rotation=45)
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig('results/task2_template_performance_1024.png')
            plt.savefig('results/task2_template_performance_1024.pdf')
            plt.close()
    
    # 图9: 热力图展示所有算法在不同优化级别下的性能
    plt.figure(figsize=(14, 8))
    
    # 收集所有优化级别上的数据
    perf_data = {}
    for opt in opt_levels:
        speedup_filename = f'results/task2_speedup_{opt}.csv'
        if os.path.exists(speedup_filename):
            # 读取CSV文件，注意没有列名
            speedup_data = pd.read_csv(speedup_filename, header=None)
            # 找到大小为1024的数据
            rows_with_1024 = speedup_data[speedup_data.iloc[:, 1] == 1024]
            if not rows_with_1024.empty:
                row = rows_with_1024.iloc[0, 2:min(9, len(speedup_data.columns))].values
                perf_data[opt] = row
    
    if perf_data:
        # 创建性能矩阵
        alg_names_short = [name[:4] + '..' if len(name) > 6 else name for name in speedup_alg_names[:len(next(iter(perf_data.values())))]]
        data_matrix = np.array([perf_data[opt] for opt in opt_levels if opt in perf_data])
        
        # 创建热力图
        cmap = LinearSegmentedColormap.from_list('speedup', ['#FFFFFF', '#D9EAD3', '#93C47D', '#6AA84F', '#38761D'])
        plt.imshow(data_matrix, cmap=cmap, aspect='auto')
        
        # 添加文本标注
        for i in range(len(data_matrix)):
            for j in range(len(data_matrix[i])):
                plt.text(j, i, f'{data_matrix[i, j]:.2f}', ha='center', va='center', 
                        color='black' if data_matrix[i, j] < 2.5 else 'white')
        
        plt.colorbar(label='Speedup')
        plt.title('Vector Sum Algorithm Speedup Heatmap Across Optimization Levels')
        plt.xlabel('Algorithm')
        plt.ylabel('Optimization Level')
        plt.yticks(range(len([opt for opt in opt_levels if opt in perf_data])), [opt for opt in opt_levels if opt in perf_data])
        plt.xticks(range(len(alg_names_short)), alg_names_short, rotation=45)
        plt.tight_layout()
        plt.savefig('results/task2_heatmap.png')
        plt.savefig('results/task2_heatmap.pdf')
        plt.close()
    
    print('All Vector Sum Charts Generated and Saved in results Directory.')


if __name__ == "__main__":
    print("Starting to Generate Vector Sum Charts...")
    plot_task2()
    print("Vector Sum Charts Generation Completed!") 