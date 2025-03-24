# 并行计算性能分析与可视化

本项目包含用于运行并行计算性能测试并生成专业可视化图表的脚本。

## 项目结构

- `task1.cpp` - 矩阵乘法实验代码（阻塞与非阻塞算法比较）
- `task2.cpp` - 向量求和实验代码（各种求和算法的比较）
- `collect_data.sh` - 数据收集脚本，自动编译不同优化级别的程序并收集结果
- `plot_results.m` - 主MATLAB绘图脚本，生成所有实验的可视化图表
- `plot_task1.m` - 用于矩阵乘法实验的单独MATLAB绘图脚本
- `plot_task2.m` - 用于向量求和实验的单独MATLAB绘图脚本

## 使用方法

### 步骤1：编译并收集实验数据

运行数据收集脚本：

```bash
chmod +x collect_data.sh
./collect_data.sh
```

这个脚本会：
1. 创建`results`目录存放结果
2. 使用不同的优化级别编译程序（O0、O1、O2、O3、Ofast）
3. 运行各版本的程序并收集结果到CSV文件

### 步骤2：生成可视化图表

复制生成的`results`目录到装有MATLAB的计算机上，然后运行：

```
matlab -nodisplay -nosplash -nodesktop -r "plot_results; exit"
```

或者在MATLAB环境中直接打开并运行：
- `plot_results.m` - 生成所有实验的图表
- `plot_task1.m` - 仅生成矩阵乘法实验的图表
- `plot_task2.m` - 仅生成向量求和实验的图表

### 生成的图表

#### 任务1（矩阵乘法）：
- `task1_execution_times.png` - 阻塞式和非阻塞式矩阵乘法在不同优化级别下的执行时间
- `task1_speedup.png` - 非阻塞相对于阻塞的加速比
- `task1_O3_comparison.png` - O3优化级别下的阻塞与非阻塞比较

#### 任务2（向量求和）：
- `task2_execution_times.png` - 不同优化级别下各算法的执行时间
- `task2_speedup.png` - 不同优化级别下各算法相对于朴素算法的加速比
- `task2_best_speedup.png` - 每个优化级别下最佳算法的加速比
- `task2_macro_template_speedup.png` - 不同优化级别下宏模板算法的加速比
- `task2_O3_execution_times.png` - O3优化级别下的执行时间详细比较
- `task2_O3_speedup.png` - O3优化级别下的加速比详细比较
- `task2_O3_best_algorithm_analysis.png` - O3级别下最佳算法分析
- `task2_template_performance_1024.png` - 纯模板算法在1024元素大小上的性能分析

## 定制化

- 修改`collect_data.sh`中的编译选项以测试其他编译参数
- 修改MATLAB脚本中的图表设置，如颜色、标记、标题等
- 在MATLAB脚本中添加新的图表类型以展示其他关注点

## 依赖

- C++编译器（g++）
- Bash shell
- MATLAB（用于生成图表） 