%% 矩阵乘法性能实验绘图工具
% 这个MATLAB脚本从收集的CSV文件中读取数据并为任务1生成高质量图表

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
colors = {'#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30'};
markers = {'o', 's', 'd', '^', 'v'};

fprintf('生成矩阵乘法 (Task 1) 图表...\n');

% 创建图形1: 在不同优化级别下的执行时间比较
figure('Position', [100, 100, 1200, 600]);

subplot(1, 2, 1);
hold on;
for i = 1:length(opt_levels)
    opt = opt_levels{i};
    filename = ['results/task1_', opt, '.csv'];
    
    if exist(filename, 'file')
        data = readtable(filename);
        plot(data.size, data.blocking, 'Color', colors{i}, 'Marker', markers{i}, 'DisplayName', ['阻塞式 - ', opt]);
    end
end
title('不同优化级别下的阻塞式矩阵乘法执行时间');
xlabel('矩阵大小');
ylabel('执行时间 (毫秒)');
legend('Location', 'northwest');
set(gca, 'XScale', 'log2');
set(gca, 'YScale', 'log');
grid on;

subplot(1, 2, 2);
hold on;
for i = 1:length(opt_levels)
    opt = opt_levels{i};
    filename = ['results/task1_', opt, '.csv'];
    
    if exist(filename, 'file')
        data = readtable(filename);
        plot(data.size, data.unblocking, 'Color', colors{i}, 'Marker', markers{i}, 'DisplayName', ['非阻塞式 - ', opt]);
    end
end
title('不同优化级别下的非阻塞式矩阵乘法执行时间');
xlabel('矩阵大小');
ylabel('执行时间 (毫秒)');
legend('Location', 'northwest');
set(gca, 'XScale', 'log2');
set(gca, 'YScale', 'log');
grid on;

% 保存图表
saveas(gcf, 'results/task1_execution_times.png');
saveas(gcf, 'results/task1_execution_times.fig');

% 创建图形2: 不同优化级别下的加速比
figure('Position', [100, 100, 900, 500]);
hold on;

for i = 1:length(opt_levels)
    opt = opt_levels{i};
    filename = ['results/task1_', opt, '.csv'];
    
    if exist(filename, 'file')
        data = readtable(filename);
        plot(data.size, data.speedup, 'Color', colors{i}, 'Marker', markers{i}, 'DisplayName', opt);
    end
end
title('不同优化级别下非阻塞相对于阻塞的加速比');
xlabel('矩阵大小');
ylabel('加速比');
legend('Location', 'best');
set(gca, 'XScale', 'log2');
grid on;

% 保存图表
saveas(gcf, 'results/task1_speedup.png');
saveas(gcf, 'results/task1_speedup.fig');

% 创建图形3: 最佳优化级别下的性能比较
figure('Position', [100, 100, 900, 500]);
hold on;

% 假设O3通常是最佳优化级别
opt = 'O3';
filename = ['results/task1_', opt, '.csv'];

if exist(filename, 'file')
    data = readtable(filename);
    plot(data.size, data.blocking, 'Color', colors{1}, 'Marker', markers{1}, 'DisplayName', '阻塞式');
    plot(data.size, data.unblocking, 'Color', colors{2}, 'Marker', markers{2}, 'DisplayName', '非阻塞式');
    
    % 添加最大和最小矩阵大小的标注
    min_size = min(data.size);
    max_size = max(data.size);
    
    min_blocking_time = data.blocking(data.size == min_size);
    min_unblocking_time = data.unblocking(data.size == min_size);
    max_blocking_time = data.blocking(data.size == max_size);
    max_unblocking_time = data.unblocking(data.size == max_size);
    
    text(min_size, min_blocking_time, [num2str(min_blocking_time, '%.2f'), ' ms'], 'VerticalAlignment', 'bottom');
    text(min_size, min_unblocking_time, [num2str(min_unblocking_time, '%.2f'), ' ms'], 'VerticalAlignment', 'bottom');
    text(max_size, max_blocking_time, [num2str(max_blocking_time, '%.2f'), ' ms'], 'VerticalAlignment', 'bottom');
    text(max_size, max_unblocking_time, [num2str(max_unblocking_time, '%.2f'), ' ms'], 'VerticalAlignment', 'bottom');
end

title(['优化级别 ', opt, ' 下的矩阵乘法性能比较']);
xlabel('矩阵大小');
ylabel('执行时间 (毫秒)');
legend('Location', 'northwest');
set(gca, 'XScale', 'log2');
set(gca, 'YScale', 'log');
grid on;

% 保存图表
saveas(gcf, 'results/task1_O3_comparison.png');
saveas(gcf, 'results/task1_O3_comparison.fig');

fprintf('所有Task 1图表已生成并保存在results目录中。\n'); 