%% 并行计算实验绘图工具
% 这个MATLAB脚本从收集的CSV文件中读取数据并生成高质量图表

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

opt_levels = {'O0', 'O1', 'O2', 'O3', 'Ofast'};
colors = {'#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F', '#000000'};
markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p'};

%% Task 1: 块矩阵乘法
fprintf('生成Task 1图表...\n');

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
figure('Position', [100, 100, 1200, 500]);
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

%% Task 2: 向量求和
fprintf('生成Task 2图表...\n');

% 获取所有算法名称
alg_names = {'朴素算法', '两路累加', '四路累加', '循环展开', '宏模板', '纯模板', '两路纯模板', '四路纯模板'};
speedup_alg_names = {'两路累加', '四路累加', '循环展开', '宏模板', '纯模板', '两路纯模板', '四路纯模板'};

% 创建图形3: O3优化级别下的执行时间比较
figure('Position', [100, 100, 1200, 600]);

for o = 1:length(opt_levels)
    opt = opt_levels{o};
    time_filename = ['results/task2_time_', opt, '.csv'];
    
    if exist(time_filename, 'file')
        time_data = readtable(time_filename);
        
        % 为每个优化级别创建一个子图
        subplot(ceil(length(opt_levels)/3), 3, o);
        hold on;
        
        % 绘制所有算法的执行时间
        for i = 2:min(9, width(time_data))
            if i == 7 && opt ~= "O0" % 只在O0级别显示纯模板算法时间
                continue;
            end
            plot(time_data.size, time_data{:,i}, 'Color', colors{i-1}, 'Marker', markers{i-1}, 'DisplayName', alg_names{i-1});
        end
        
        title(['优化级别 ', opt, ' 下的执行时间']);
        xlabel('向量大小');
        ylabel('执行时间 (微秒)');
        legend('Location', 'northwest');
        set(gca, 'XScale', 'log2');
        set(gca, 'YScale', 'log');
        grid on;
    end
end

% 保存图表
saveas(gcf, 'results/task2_execution_times.png');
saveas(gcf, 'results/task2_execution_times.fig');

% 创建图形4: 各算法相对于朴素算法的加速比
figure('Position', [100, 100, 1200, 600]);

% O3优化级别加速比的单独图表
for o = 1:length(opt_levels)
    opt = opt_levels{o};
    speedup_filename = ['results/task2_speedup_', opt, '.csv'];
    
    if exist(speedup_filename, 'file')
        speedup_data = readtable(speedup_filename);
        
        % 为每个优化级别创建一个子图
        subplot(ceil(length(opt_levels)/3), 3, o);
        hold on;
        
        % 绘制所有算法的加速比
        for i = 2:min(8, width(speedup_data))
            if i >= 6 && opt ~= "O0" % 只在O0级别显示纯模板算法加速比
                continue;
            end
            plot(speedup_data.size, speedup_data{:,i}, 'Color', colors{i}, 'Marker', markers{i}, 'DisplayName', speedup_alg_names{i-1});
        end
        
        title(['优化级别 ', opt, ' 下的加速比']);
        xlabel('向量大小');
        ylabel('相对于朴素算法的加速比');
        legend('Location', 'best');
        set(gca, 'XScale', 'log2');
        grid on;
    end
end

% 保存图表
saveas(gcf, 'results/task2_speedup.png');
saveas(gcf, 'results/task2_speedup.fig');

% 创建图5: 不同优化级别下的最佳算法比较
figure('Position', [100, 100, 900, 500]);
hold on;

% 对每个优化级别，找出最佳算法并绘制
best_speedups = cell(length(opt_levels), 1);
for i = 1:length(opt_levels)
    opt = opt_levels{i};
    speedup_filename = ['results/task2_speedup_', opt, '.csv'];
    
    if exist(speedup_filename, 'file')
        speedup_data = readtable(speedup_filename);
        
        % 找出每个矩阵大小下的最佳加速比
        all_speedups = speedup_data{:, 2:end};
        [best_speedup, ~] = max(all_speedups, [], 2);
        best_speedups{i} = best_speedup;
        
        % 绘制最佳加速比
        plot(speedup_data.size, best_speedup, 'Color', colors{i}, 'Marker', markers{i}, 'DisplayName', opt);
    end
end

title('不同优化级别下的最佳算法加速比');
xlabel('向量大小');
ylabel('最佳加速比');
legend('Location', 'best');
set(gca, 'XScale', 'log2');
grid on;

% 保存图表
saveas(gcf, 'results/task2_best_speedup.png');
saveas(gcf, 'results/task2_best_speedup.fig');

% 创建图6: 比较所有优化级别下的宏模板算法
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
saveas(gcf, 'results/task2_macro_template_speedup.png');
saveas(gcf, 'results/task2_macro_template_speedup.fig');

fprintf('所有图表已生成并保存在results目录中。\n'); 