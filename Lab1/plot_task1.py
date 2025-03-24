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

def plot_task1():
    """生成矩阵乘法(Task 1)的图表"""
    print('生成矩阵乘法 (Task 1) 图表...')
    
    # 图1: 在不同优化级别下的执行时间比较
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 阻塞式矩阵乘法
    for i, opt in enumerate(opt_levels):
        filename = f'results/task1_{opt}.csv'
        if os.path.exists(filename):
            data = pd.read_csv(filename)
            axes[0].plot(data['size'], data['blocking'], color=colors[i], marker=markers[i], 
                         label=f'阻塞式 - {opt}')
    
    axes[0].set_title('不同优化级别下的阻塞式矩阵乘法执行时间')
    axes[0].set_xlabel('矩阵大小')
    axes[0].set_ylabel('执行时间 (毫秒)')
    axes[0].set_xscale('log', base=2)
    axes[0].set_yscale('log')
    axes[0].legend(loc='upper left')
    axes[0].grid(True)
    
    # 非阻塞式矩阵乘法
    for i, opt in enumerate(opt_levels):
        filename = f'results/task1_{opt}.csv'
        if os.path.exists(filename):
            data = pd.read_csv(filename)
            axes[1].plot(data['size'], data['unblocking'], color=colors[i], marker=markers[i], 
                         label=f'非阻塞式 - {opt}')
    
    axes[1].set_title('不同优化级别下的非阻塞式矩阵乘法执行时间')
    axes[1].set_xlabel('矩阵大小')
    axes[1].set_ylabel('执行时间 (毫秒)')
    axes[1].set_xscale('log', base=2)
    axes[1].set_yscale('log')
    axes[1].legend(loc='upper left')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/task1_execution_times.png')
    plt.savefig('results/task1_execution_times.pdf')  # 同时保存PDF格式，以便在论文中使用
    plt.close()
    
    # 图2: 不同优化级别下的加速比
    plt.figure(figsize=(10, 6))
    
    for i, opt in enumerate(opt_levels):
        filename = f'results/task1_{opt}.csv'
        if os.path.exists(filename):
            data = pd.read_csv(filename)
            plt.plot(data['size'], data['speedup'], color=colors[i], marker=markers[i], label=opt)
    
    plt.title('不同优化级别下非阻塞相对于阻塞的加速比')
    plt.xlabel('矩阵大小')
    plt.ylabel('加速比')
    plt.xscale('log', base=2)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/task1_speedup.png')
    plt.savefig('results/task1_speedup.pdf')
    plt.close()
    
    # 图3: 最佳优化级别下的性能比较
    plt.figure(figsize=(10, 6))
    
    # 假设O3通常是最佳优化级别
    opt = 'O3'
    filename = f'results/task1_{opt}.csv'
    
    if os.path.exists(filename):
        data = pd.read_csv(filename)
        plt.plot(data['size'], data['blocking'], color=colors[0], marker=markers[0], label='阻塞式')
        plt.plot(data['size'], data['unblocking'], color=colors[1], marker=markers[1], label='非阻塞式')
        
        # 添加最大和最小矩阵大小的标注
        min_size = data['size'].min()
        max_size = data['size'].max()
        
        min_blocking_idx = data['size'].idxmin()
        min_unblocking_idx = data['size'].idxmin()
        max_blocking_idx = data['size'].idxmax()
        max_unblocking_idx = data['size'].idxmax()
        
        min_blocking_time = data.loc[min_blocking_idx, 'blocking']
        min_unblocking_time = data.loc[min_unblocking_idx, 'unblocking']
        max_blocking_time = data.loc[max_blocking_idx, 'blocking']
        max_unblocking_time = data.loc[max_unblocking_idx, 'unblocking']
        
        plt.text(min_size, min_blocking_time, f'{min_blocking_time:.2f} ms', va='bottom')
        plt.text(min_size, min_unblocking_time, f'{min_unblocking_time:.2f} ms', va='bottom')
        plt.text(max_size, max_blocking_time, f'{max_blocking_time:.2f} ms', va='bottom')
        plt.text(max_size, max_unblocking_time, f'{max_unblocking_time:.2f} ms', va='bottom')
    
    plt.title(f'优化级别 {opt} 下的矩阵乘法性能比较')
    plt.xlabel('矩阵大小')
    plt.ylabel('执行时间 (毫秒)')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/task1_O3_comparison.png')
    plt.savefig('results/task1_O3_comparison.pdf')
    plt.close()
    
    # 图4: 所有优化级别的加速比分布
    plt.figure(figsize=(10, 6))
    
    speedups = []
    sizes = []
    opt_labels = []
    
    for opt in opt_levels:
        filename = f'results/task1_{opt}.csv'
        if os.path.exists(filename):
            data = pd.read_csv(filename)
            speedups.extend(data['speedup'].tolist())
            sizes.extend(data['size'].tolist())
            opt_labels.extend([opt] * len(data))
    
    # 使用箱线图展示不同优化级别的加速比分布
    if speedups:
        df = pd.DataFrame({'优化级别': opt_labels, '加速比': speedups, '矩阵大小': sizes})
        box_data = [df[df['优化级别'] == opt]['加速比'] for opt in opt_levels if not df[df['优化级别'] == opt].empty]
        
        plt.boxplot(box_data, labels=[opt for opt in opt_levels if not df[df['优化级别'] == opt].empty])
        plt.title('不同优化级别下的加速比分布')
        plt.xlabel('优化级别')
        plt.ylabel('加速比')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/task1_speedup_distribution.png')
        plt.savefig('results/task1_speedup_distribution.pdf')
        plt.close()
    
    print('所有矩阵乘法图表已生成并保存在results目录中。')


if __name__ == "__main__":
    print("开始生成矩阵乘法图表...")
    plot_task1()
    print("矩阵乘法图表生成完成!") 