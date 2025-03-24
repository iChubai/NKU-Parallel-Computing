#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
矩阵乘法性能实验Python绘图工具
此脚本从收集的CSV文件中读取数据并生成矩阵乘法(task1)的高质量图表
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
alg_names = ['Naive', 'Loop Unrolling', 'Blocking', 'Strassen', 'Special Opt', 'Template', 'Pure Template']
speedup_alg_names = ['Loop Unrolling', 'Blocking', 'Strassen', 'Special Opt', 'Template', 'Pure Template']

# 导出变量供其他模块使用
task1_colors = colors
task1_markers = markers
task1_alg_names = alg_names
task1_speedup_alg_names = speedup_alg_names

def plot_task1():
    """生成矩阵乘法(Task 1)的图表"""
    print('生成矩阵乘法 (Task 1) 图表...')
    
    # 图1: 不同优化级别下的执行时间比较
    fig = plt.figure(figsize=(15, 12))
    
    for o, opt in enumerate(opt_levels):
        time_filename = f'results/task1_time_{opt}.csv'
        if os.path.exists(time_filename):
            # 读取CSV文件，注意没有列名，需要自己提供
            time_data = pd.read_csv(time_filename, header=None)
            print(f"读取文件: {time_filename}, 形状: {time_data.shape}")
            
            # 为每个优化级别创建一个子图
            ax = fig.add_subplot(2, 3, o+1)
            
            # 使用第1列作为矩阵大小(索引0是行号，索引1是第一个数据列)，绘制所有算法的执行时间
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
            ax.set_xlabel('Matrix Size (n*n)')
            ax.set_ylabel('Execution Time (ms)')
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.legend(loc='upper left')
            ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/task1_execution_times.png')
    plt.savefig('results/task1_execution_times.pdf')
    plt.close()
    
    # 图2: 不同优化级别下的加速比
    fig = plt.figure(figsize=(15, 12))
    
    for o, opt in enumerate(opt_levels):
        speedup_filename = f'results/task1_speedup_{opt}.csv'
        if os.path.exists(speedup_filename):
            # 读取CSV文件，注意没有列名，需要自己提供
            speedup_data = pd.read_csv(speedup_filename, header=None)
            print(f"读取文件: {speedup_filename}, 形状: {speedup_data.shape}")
            
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
            ax.set_xlabel('Matrix Size (n*n)')
            ax.set_ylabel('Speedup Relative to Naive Algorithm')
            ax.set_xscale('log', base=2)
            ax.legend(loc='best')
            ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/task1_speedup.png')
    plt.savefig('results/task1_speedup.pdf')
    plt.close()
    
    # 图3: 不同优化级别下的最佳算法比较
    plt.figure(figsize=(10, 6))
    
    for i, opt in enumerate(opt_levels):
        speedup_filename = f'results/task1_speedup_{opt}.csv'
        if os.path.exists(speedup_filename):
            # 读取CSV文件，注意没有列名
            speedup_data = pd.read_csv(speedup_filename, header=None)
            
            # 找出每个矩阵大小下的最佳加速比
            best_speedup = speedup_data.iloc[:, 2:].max(axis=1)
            plt.plot(speedup_data.iloc[:, 1], best_speedup, color=colors[i], marker=markers[i], label=opt)
    
    plt.title('Best Algorithm Speedup Across Optimization Levels')
    plt.xlabel('Matrix Size (n*n)')
    plt.ylabel('Best Speedup')
    plt.xscale('log', base=2)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/task1_best_speedup.png')
    plt.savefig('results/task1_best_speedup.pdf')
    plt.close()
    
    # 图4: 比较所有优化级别下的Strassen算法
    plt.figure(figsize=(10, 6))
    
    for i, opt in enumerate(opt_levels):
        speedup_filename = f'results/task1_speedup_{opt}.csv'
        if os.path.exists(speedup_filename):
            # 读取CSV文件，注意没有列名
            speedup_data = pd.read_csv(speedup_filename, header=None)
            # Strassen算法通常是第4列
            if len(speedup_data.columns) >= 5:
                plt.plot(speedup_data.iloc[:, 1], speedup_data.iloc[:, 4], color=colors[i], marker=markers[i], label=opt)
    
    plt.title('Strassen Algorithm Speedup Across Optimization Levels')
    plt.xlabel('Matrix Size (n*n)')
    plt.ylabel('Speedup Relative to Naive Algorithm')
    plt.xscale('log', base=2)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/task1_strassen_speedup.png')
    plt.savefig('results/task1_strassen_speedup.pdf')
    plt.close()
    
    # 图5: O3优化级别下的执行时间详细比较
    plt.figure(figsize=(10, 6))
    
    opt = 'O3'
    time_filename = f'results/task1_time_{opt}.csv'
    if os.path.exists(time_filename):
        # 读取CSV文件，注意没有列名
        time_data = pd.read_csv(time_filename, header=None)
        
        for i in range(2, min(7, len(time_data.columns))):
            plt.plot(time_data.iloc[:, 1], time_data.iloc[:, i], color=colors[i-2], marker=markers[i-2], 
                    label=alg_names[i-2])
        
        plt.title(f'Optimization Level {opt} Matrix Multiplication Execution Time')
        plt.xlabel('Matrix Size (n*n)')
        plt.ylabel('Execution Time (ms)')
        plt.xscale('log', base=2)
        plt.yscale('log')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/task1_O3_execution_times.png')
        plt.savefig('results/task1_O3_execution_times.pdf')
        plt.close()
    
    # 图6: O3优化级别下的加速比详细比较
    plt.figure(figsize=(10, 6))
    
    opt = 'O3'
    speedup_filename = f'results/task1_speedup_{opt}.csv'
    if os.path.exists(speedup_filename):
        # 读取CSV文件，注意没有列名
        speedup_data = pd.read_csv(speedup_filename, header=None)
        
        for i in range(2, min(7, len(speedup_data.columns))):
            plt.plot(speedup_data.iloc[:, 1], speedup_data.iloc[:, i], color=colors[i-1], marker=markers[i-1], 
                    label=speedup_alg_names[i-2])
        
        plt.title(f'Optimization Level {opt} Matrix Multiplication Speedup')
        plt.xlabel('Matrix Size (n*n)')
        plt.ylabel('Speedup Relative to Naive Algorithm')
        plt.xscale('log', base=2)
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/task1_O3_speedup.png')
        plt.savefig('results/task1_O3_speedup.pdf')
        plt.close()
    
    # 图7: 分析矩阵大小对各算法性能的影响
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    opt = 'O3'
    time_filename = f'results/task1_time_{opt}.csv'
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
        
        axes[0].set_title(f'Best Algorithm Distribution at Optimization Level {opt}')
        axes[0].set_xlabel('Matrix Size (n*n)')
        axes[0].set_yticks(range(len(unique_best)))
        axes[0].set_yticklabels([alg_names[idx] for idx in unique_best])
        axes[0].set_xscale('log', base=2)
        axes[0].grid(True)
        axes[0].legend(loc='best')
    
    speedup_filename = f'results/task1_speedup_{opt}.csv'
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
        
        axes[1].set_title(f'Maximum Speedup at Optimization Level {opt}')
        axes[1].set_xlabel('Matrix Size (n*n)')
        axes[1].set_ylabel('Speedup Relative to Naive Algorithm')
        axes[1].set_xscale('log', base=2)
        axes[1].grid(True)
        axes[1].legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('results/task1_O3_best_algorithm_analysis.png')
    plt.savefig('results/task1_O3_best_algorithm_analysis.pdf')
    plt.close()
    
    # 图8: 矩阵分块对性能的影响
    plt.figure(figsize=(10, 6))
    
    for o, opt in enumerate(['O0', 'O3']):
        time_filename = f'results/task1_time_{opt}.csv'
        if os.path.exists(time_filename):
            # 读取CSV文件，注意没有列名
            time_data = pd.read_csv(time_filename, header=None)
            
            # 分块算法通常是第3列
            if len(time_data.columns) >= 4:
                block_time = time_data.iloc[:, 3]  # 分块算法时间
                naive_time = time_data.iloc[:, 2]  # 朴素算法时间
                speedup = naive_time / block_time
                plt.plot(time_data.iloc[:, 1], speedup, color=colors[o], marker=markers[o], 
                        label=f'{opt} Optimization Level')
    
    plt.title('Blocking Algorithm Speedup Relative to Naive Algorithm')
    plt.xlabel('Matrix Size (n*n)')
    plt.ylabel('Speedup')
    plt.xscale('log', base=2)
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('results/task1_blocking_analysis.png')
    plt.savefig('results/task1_blocking_analysis.pdf')
    plt.close()
    
    # 图9: 分析特定大小矩阵下的性能
    plt.figure(figsize=(10, 6))
    
    opt = 'O3'
    time_filename = f'results/task1_time_{opt}.csv'
    if os.path.exists(time_filename):
        # 读取CSV文件，注意没有列名
        time_data = pd.read_csv(time_filename, header=None)
        
        # 选择特定大小的矩阵
        selected_sizes = [128, 512, 1024]
        for s, size in enumerate(selected_sizes):
            # 找到最接近的大小
            closest_idx = np.abs(time_data.iloc[:, 1] - size).argmin()
            actual_size = time_data.iloc[closest_idx, 1]
            
            # 提取该大小下的所有算法时间
            times = time_data.iloc[closest_idx, 2:7].values
            
            # 对性能进行排序
            sorted_indices = np.argsort(times)
            sorted_times = times[sorted_indices]
            sorted_names = [alg_names[i] for i in sorted_indices]
            
            # 绘制柱状图
            ax = plt.subplot(1, len(selected_sizes), s+1)
            bars = ax.bar(range(len(sorted_times)), sorted_times, color=colors[:len(sorted_times)])
            
            # 添加标签
            ax.set_title(f'Matrix Size {actual_size}x{actual_size}')
            ax.set_ylabel('Execution Time (ms)' if s == 0 else '')
            ax.set_xticks(range(len(sorted_names)))
            ax.set_xticklabels(sorted_names, rotation=45)
            
            # 添加性能标签
            for i, v in enumerate(sorted_times):
                ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/task1_size_specific_performance.png')
    plt.savefig('results/task1_size_specific_performance.pdf')
    plt.close()
    
    # 图10: 矩阵乘法算法的加速比箱线图分析
    plt.figure(figsize=(12, 7))
    
    speedup_data_by_alg = {}
    for opt in opt_levels:
        speedup_filename = f'results/task1_speedup_{opt}.csv'
        if os.path.exists(speedup_filename):
            # 读取CSV文件，注意没有列名
            data = pd.read_csv(speedup_filename, header=None)
            
            # 对每种算法收集加速比数据
            num_cols = min(7, len(data.columns))
            for i in range(2, num_cols):
                alg_name = speedup_alg_names[i-2]
                if alg_name not in speedup_data_by_alg:
                    speedup_data_by_alg[alg_name] = []
                speedup_data_by_alg[alg_name].append(data.iloc[:, i])
    
    # 合并数据并创建箱线图
    box_data = []
    labels = []
    for alg, values in speedup_data_by_alg.items():
        # 合并来自不同优化级别的数据
        box_data.append(pd.concat(values))
        labels.append(alg)
    
    if box_data:
        plt.boxplot(box_data, labels=labels, showfliers=False, patch_artist=True,
                  boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))
        plt.title('Matrix Multiplication Speedup Distribution')
        plt.ylabel('Speedup')
        plt.grid(True, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/task1_speedup_boxplot.png')
        plt.savefig('results/task1_speedup_boxplot.pdf')
        plt.close()
    
    # 图11: 创建热力图展示所有优化级别下算法在特定大小矩阵上的性能对比
    plt.figure(figsize=(12, 7))
    
    target_size = 512  # 目标矩阵大小
    perf_data = {}
    
    for opt in opt_levels:
        speedup_filename = f'results/task1_speedup_{opt}.csv'
        if os.path.exists(speedup_filename):
            # 读取CSV文件，注意没有列名
            data = pd.read_csv(speedup_filename, header=None)
            
            # 找到最接近目标大小的行
            closest_idx = np.abs(data.iloc[:, 1] - target_size).argmin()
            perf_data[opt] = data.iloc[closest_idx, 2:min(7, len(data.columns))].values
    
    if perf_data:
        # 创建热力图数据矩阵
        matrix_data = np.array(list(perf_data.values()))
        
        # 创建热力图
        plt.imshow(matrix_data, aspect='auto', cmap='viridis')
        plt.colorbar(label='Speedup')
        plt.xlabel('Algorithm')
        plt.ylabel('Optimization Level')
        plt.title(f'{target_size}x{target_size} Matrix Algorithm Performance Heatmap')
        
        # 设置坐标轴标签
        plt.yticks(range(len(opt_levels)), opt_levels)
        alg_names_short = [name[:10] for name in speedup_alg_names[:matrix_data.shape[1]]]
        plt.xticks(range(len(alg_names_short)), alg_names_short, rotation=45)
        
        # 添加数值标注
        for i in range(matrix_data.shape[0]):
            for j in range(matrix_data.shape[1]):
                plt.text(j, i, f'{matrix_data[i, j]:.2f}', ha='center', va='center', 
                        color='white' if matrix_data[i, j] < np.max(matrix_data)/2 else 'black')
        
        plt.tight_layout()
        plt.savefig('results/task1_heatmap.png')
        plt.savefig('results/task1_heatmap.pdf')
        plt.close()
    
    print('All Matrix Multiplication Charts Generated and Saved in results Directory.')


if __name__ == "__main__":
    print("Starting to Generate Matrix Multiplication Charts...")
    plot_task1()
    print("Matrix Multiplication Charts Generation Completed!") 