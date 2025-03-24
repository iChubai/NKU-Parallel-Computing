%% 向量求和性能实验绘图工具
% 这个MATLAB脚本从收集的CSV文件中读取数据并为任务2生成高质量图表

clear;
close all;
clc;

% 设置图表样式
set(0, 'DefaultAxesFontSize', 12);
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesLineWidth', 1.5);
set(0, 'DefaultAxesBox', 'on');
set(0, 'DefaultFigureColor', 'white');

% 优化级别和图形样式
opt_levels = {'O0', 'O1', 'O2', 'O3', 'Ofast'};
colors = {'#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F', '#000000'};
markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p'};
alg_names = {'朴素算法', '两路累加', '四路累加', '循环展开', '宏模板', '纯模板', '两路纯模板', '四路纯模板'};
speedup_alg_names = {'两路累加', '四路累加', '循环展开', '宏模板', '纯模板', '两路纯模板', '四路纯模板'};

fprintf('生成向量求和 (Task 2) 图表...\n');

% 创建图1: O3优化级别下的时间比较
figure('Position', [100, 100, 1000, 600]);
hold on;

opt = 'O3'; % 以O3优化级别为例
time_filename = ['results/task2_time_', opt, '.csv'];

if exist(time_filename, 'file')
    time_data = readtable(time_filename);
    
    % 绘制各算法的执行时间
    for i = 2:min(6, width(time_data)) % 排除纯模板算法（通常只在1024大小上有数据）
        plot(time_data.size, time_data{:,i}, 'Color', colors{i-1}, 'Marker', markers{i-1}, 'DisplayName', alg_names{i-1});
    end
    
    title(['优化级别 ', opt, ' 下的向量求和执行时间']);
    xlabel('向量大小');
    ylabel('执行时间 (微秒)');
    legend('Location', 'northwest');
    set(gca, 'XScale', 'log2');
    set(gca, 'YScale', 'log');
    grid on;
    
    % 保存图表
    saveas(gcf, 'results/task2_O3_execution_times.png');
    saveas(gcf, 'results/task2_O3_execution_times.fig');
end

% 创建图2: O3优化级别下的加速比
figure('Position', [100, 100, 1000, 600]);
hold on;

opt = 'O3'; % 以O3优化级别为例
speedup_filename = ['results/task2_speedup_', opt, '.csv'];

if exist(speedup_filename, 'file')
    speedup_data = readtable(speedup_filename);
    
    % 绘制各算法的加速比
    for i = 2:min(6, width(speedup_data)) % 排除纯模板算法
        plot(speedup_data.size, speedup_data{:,i}, 'Color', colors{i}, 'Marker', markers{i}, 'DisplayName', speedup_alg_names{i-1});
    end
    
    title(['优化级别 ', opt, ' 下的向量求和加速比']);
    xlabel('向量大小');
    ylabel('相对于朴素算法的加速比');
    legend('Location', 'best');
    set(gca, 'XScale', 'log2');
    grid on;
    
    % 保存图表
    saveas(gcf, 'results/task2_O3_speedup.png');
    saveas(gcf, 'results/task2_O3_speedup.fig');
end

% 创建图3: 不同优化级别下的宏模板算法比较
figure('Position', [100, 100, 900, 500]);
hold on;

for i = 1:length(opt_levels)
    opt = opt_levels{i};
    speedup_filename = ['results/task2_speedup_', opt, '.csv'];
    
    if exist(speedup_filename, 'file')
        speedup_data = readtable(speedup_filename);
        % 宏模板算法通常是第5列
        if width(speedup_data) >= 5
            plot(speedup_data.size, speedup_data{:,5}, 'Color', colors{i}, 'Marker', markers{i}, 'DisplayName', opt);
        end
    end
end

title('不同优化级别下宏模板算法的加速比');
xlabel('向量大小');
ylabel('相对于朴素算法的加速比');
legend('Location', 'best');
set(gca, 'XScale', 'log2');
grid on;

% 保存图表
saveas(gcf, 'results/task2_macro_template_comparison.png');
saveas(gcf, 'results/task2_macro_template_comparison.fig');

% 创建图4: 分析最佳算法随向量大小的变化
figure('Position', [100, 100, 1000, 600]);

% 创建两个子图
subplot(1, 2, 1);
hold on;

opt = 'O3';
time_filename = ['results/task2_time_', opt, '.csv'];
if exist(time_filename, 'file')
    time_data = readtable(time_filename);
    time_values = time_data{:, 2:min(6, width(time_data))};
    [~, best_indices] = min(time_values, [], 2);
    
    % 为每种最佳算法选择不同颜色
    unique_best = unique(best_indices);
    for i = 1:length(unique_best)
        alg_idx = unique_best(i);
        sizes = time_data.size(best_indices == alg_idx);
        scatter(sizes, ones(size(sizes))*i, 50, colors{alg_idx}, 'filled', 'DisplayName', alg_names{alg_idx});
    end
    
    title(['优化级别 ', opt, ' 下最快算法的分布']);
    xlabel('向量大小');
    yticks(1:length(unique_best));
    yticklabels(alg_names(unique_best));
    set(gca, 'XScale', 'log2');
    grid on;
    legend('Location', 'best');
end

subplot(1, 2, 2);
hold on;

opt = 'O3';
speedup_filename = ['results/task2_speedup_', opt, '.csv'];
if exist(speedup_filename, 'file')
    speedup_data = readtable(speedup_filename);
    
    % 寻找最大加速比
    [max_speedup, ~] = max(speedup_data{:, 2:min(6, width(speedup_data))}, [], 2);
    plot(speedup_data.size, max_speedup, 'k-', 'LineWidth', 2, 'DisplayName', '最大加速比');
    
    % 在图上标注最佳点
    [peak_speedup, peak_idx] = max(max_speedup);
    peak_size = speedup_data.size(peak_idx);
    scatter(peak_size, peak_speedup, 100, 'r', 'filled', 'DisplayName', '峰值加速比');
    text(peak_size, peak_speedup, [' ' num2str(peak_speedup, '%.2f'), 'x'], 'VerticalAlignment', 'bottom');
    
    title(['优化级别 ', opt, ' 下的最大加速比']);
    xlabel('向量大小');
    ylabel('相对于朴素算法的加速比');
    set(gca, 'XScale', 'log2');
    grid on;
    legend('Location', 'best');
end

% 保存图表
saveas(gcf, 'results/task2_O3_best_algorithm_analysis.png');
saveas(gcf, 'results/task2_O3_best_algorithm_analysis.fig');

% 创建图5：纯模板算法在1024元素时的性能分析（针对O0优化级别）
figure('Position', [100, 100, 900, 500]);
hold on;

opt = 'O0'; % 纯模板算法通常在O0级别最有优势
time_filename = ['results/task2_time_', opt, '.csv'];
if exist(time_filename, 'file')
    time_data = readtable(time_filename);
    small_data = time_data(time_data.size == 1024, :);
    
    if ~isempty(small_data)
        % 提取所有算法的时间数据
        times = small_data{1, 2:end};
        
        % 创建条形图
        bar(times);
        xticks(1:length(alg_names));
        xticklabels(alg_names);
        xtickangle(45);
        
        % 在每个条形上标出具体数值
        for i = 1:length(times)
            text(i, times(i), [num2str(times(i), '%.2f')], 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
        
        title(['向量大小1024的性能比较 (', opt, ' 优化)']);
        ylabel('执行时间 (微秒)');
        grid on;
    end
end

% 保存图表
saveas(gcf, 'results/task2_template_performance_1024.png');
saveas(gcf, 'results/task2_template_performance_1024.fig');

fprintf('所有Task 2图表已生成并保存在results目录中。\n'); 