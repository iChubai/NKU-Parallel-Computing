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
colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F', '#000000']
markers = ['o', 's', 'd', '^', 'v', '>', '<', 'p']
alg_names = ['朴素算法', '两路累加', '四路累加', '循环展开', '宏模板', '纯模板', '两路纯模板', '四路纯模板']
speedup_alg_names = ['两路累加', '四路累加', '循环展开', '宏模板', '纯模板', '两路纯模板', '四路纯模板']

def plot_task2():
    """生成向量求和(Task 2)的图表"""
    print('生成向量求和 (Task 2) 图表...')
    
    # 图1: 不同优化级别下的执行时间比较
    fig = plt.figure(figsize=(15, 12))
    
    for o, opt in enumerate(opt_levels):
        time_filename = f'results/task2_time_{opt}.csv'
        if os.path.exists(time_filename):
            time_data = pd.read_csv(time_filename)
            
            # 为每个优化级别创建一个子图
            ax = fig.add_subplot(2, 3, o+1)
            
            # 绘制所有算法的执行时间
            for i in range(1, min(8, len(time_data.columns))):
                if i == 6 and opt != 'O0':  # 只在O0级别显示纯模板算法时间
                    continue
                ax.plot(time_data['size'], time_data.iloc[:, i], color=colors[i-1], marker=markers[i-1], 
                       label=alg_names[i-1])
            
            ax.set_title(f'优化级别 {opt} 下的执行时间')
            ax.set_xlabel('向量大小')
            ax.set_ylabel('执行时间 (微秒)')
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
            speedup_data = pd.read_csv(speedup_filename)
            
            # 为每个优化级别创建一个子图
            ax = fig.add_subplot(2, 3, o+1)
            
            # 绘制所有算法的加速比
            for i in range(1, min(8, len(speedup_data.columns))):
                if i >= 5 and opt != 'O0':  # 只在O0级别显示纯模板算法加速比
                    continue
                ax.plot(speedup_data['size'], speedup_data.iloc[:, i], color=colors[i], marker=markers[i], 
                       label=speedup_alg_names[i-1])
            
            ax.set_title(f'优化级别 {opt} 下的加速比')
            ax.set_xlabel('向量大小')
            ax.set_ylabel('相对于朴素算法的加速比')
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
            speedup_data = pd.read_csv(speedup_filename)
            
            # 找出每个向量大小下的最佳加速比
            best_speedup = speedup_data.iloc[:, 1:].max(axis=1)
            plt.plot(speedup_data['size'], best_speedup, color=colors[i], marker=markers[i], label=opt)
    
    plt.title('不同优化级别下的最佳算法加速比')
    plt.xlabel('向量大小')
    plt.ylabel('最佳加速比')
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
            speedup_data = pd.read_csv(speedup_filename)
            # 宏模板算法通常是第5列
            if len(speedup_data.columns) >= 5:
                plt.plot(speedup_data['size'], speedup_data.iloc[:, 4], color=colors[i], marker=markers[i], label=opt)
    
    plt.title('不同优化级别下宏模板算法的加速比')
    plt.xlabel('向量大小')
    plt.ylabel('相对于朴素算法的加速比')
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
        time_data = pd.read_csv(time_filename)
        
        for i in range(1, min(6, len(time_data.columns))):
            plt.plot(time_data['size'], time_data.iloc[:, i], color=colors[i-1], marker=markers[i-1], 
                    label=alg_names[i-1])
        
        plt.title(f'优化级别 {opt} 下的向量求和执行时间')
        plt.xlabel('向量大小')
        plt.ylabel('执行时间 (微秒)')
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
        speedup_data = pd.read_csv(speedup_filename)
        
        for i in range(1, min(6, len(speedup_data.columns))):
            plt.plot(speedup_data['size'], speedup_data.iloc[:, i], color=colors[i], marker=markers[i], 
                    label=speedup_alg_names[i-1])
        
        plt.title(f'优化级别 {opt} 下的向量求和加速比')
        plt.xlabel('向量大小')
        plt.ylabel('相对于朴素算法的加速比')
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
        time_data = pd.read_csv(time_filename)
        time_values = time_data.iloc[:, 1:min(6, len(time_data.columns))].values
        best_indices = np.argmin(time_values, axis=1)
        
        # 为每种最佳算法选择不同颜色
        unique_best = np.unique(best_indices)
        for i, alg_idx in enumerate(unique_best):
            sizes = time_data.loc[best_indices == alg_idx, 'size'].values
            axes[0].scatter(sizes, np.ones(len(sizes))*i, s=100, color=colors[alg_idx], label=alg_names[alg_idx])
        
        axes[0].set_title(f'优化级别 {opt} 下最快算法的分布')
        axes[0].set_xlabel('向量大小')
        axes[0].set_yticks(range(len(unique_best)))
        axes[0].set_yticklabels([alg_names[idx] for idx in unique_best])
        axes[0].set_xscale('log', base=2)
        axes[0].grid(True)
        axes[0].legend(loc='best')
    
    speedup_filename = f'results/task2_speedup_{opt}.csv'
    if os.path.exists(speedup_filename):
        speedup_data = pd.read_csv(speedup_filename)
        
        # 寻找最大加速比
        max_speedup = speedup_data.iloc[:, 1:min(6, len(speedup_data.columns))].max(axis=1)
        axes[1].plot(speedup_data['size'], max_speedup, 'k-', linewidth=2, label='最大加速比')
        
        # 在图上标注最佳点
        peak_idx = max_speedup.idxmax()
        peak_speedup = max_speedup[peak_idx]
        peak_size = speedup_data.loc[peak_idx, 'size']
        axes[1].scatter([peak_size], [peak_speedup], s=100, c='r', label='峰值加速比')
        axes[1].text(peak_size, peak_speedup, f' {peak_speedup:.2f}x', va='bottom')
        
        axes[1].set_title(f'优化级别 {opt} 下的最大加速比')
        axes[1].set_xlabel('向量大小')
        axes[1].set_ylabel('相对于朴素算法的加速比')
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
        time_data = pd.read_csv(time_filename)
        small_data = time_data[time_data['size'] == 1024]
        
        if not small_data.empty:
            # 提取所有算法的时间数据
            times = small_data.iloc[0, 1:].values
            
            # 创建条形图
            bars = plt.bar(range(len(times)), times, color=colors[:len(times)])
            
            # 在每个条形上标出具体数值
            for i, v in enumerate(times):
                plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
            
            plt.title(f'向量大小1024的性能比较 ({opt} 优化)')
            plt.ylabel('执行时间 (微秒)')
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
            speedup_data = pd.read_csv(speedup_filename)
            if '1024' in speedup_data['size'].values:
                row = speedup_data[speedup_data['size'] == 1024].iloc[0, 1:min(8, len(speedup_data.columns))].values
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
        
        plt.colorbar(label='加速比')
        plt.title('不同优化级别下向量求和算法在1024大小上的加速比热力图')
        plt.xlabel('算法')
        plt.ylabel('优化级别')
        plt.yticks(range(len([opt for opt in opt_levels if opt in perf_data])), [opt for opt in opt_levels if opt in perf_data])
        plt.xticks(range(len(alg_names_short)), alg_names_short, rotation=45)
        plt.tight_layout()
        plt.savefig('results/task2_heatmap.png')
        plt.savefig('results/task2_heatmap.pdf')
        plt.close()
    
    print('所有向量求和图表已生成并保存在results目录中。')


if __name__ == "__main__":
    print("开始生成向量求和图表...")
    plot_task2()
    print("向量求和图表生成完成!") 