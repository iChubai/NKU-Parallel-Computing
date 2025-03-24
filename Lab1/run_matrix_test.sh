#!/bin/bash

# 创建输出目录
mkdir -p results
mkdir -p results/perf_results

# 检查是否安装了perf
if command -v perf &> /dev/null; then
    HAS_PERF=true
    echo "perf工具已安装，将进行性能分析"
else
    HAS_PERF=false
    echo "警告：找不到perf命令，将只使用chrono分析"
    echo "您可以使用 'sudo apt-get install linux-tools-common linux-tools-generic' 安装perf"
    echo "或 'sudo apt-get install linux-tools-$(uname -r)' 安装适合当前内核的工具"
fi

# 编译优化的矩阵测试程序
echo "编译矩阵乘法测试程序..."
g++ -fopenmp -O3 -march=native task1.cpp -o task1_matrix_test

# 设置线程数量（根据系统核心数调整）
CORE_COUNT=$(nproc)
export OMP_NUM_THREADS=$CORE_COUNT
echo "设置OpenMP线程数为: $CORE_COUNT"

# 运行主要测试
echo "运行矩阵乘法测试..."
./task1_matrix_test

# 如果安装了perf，则运行性能分析
if [ "$HAS_PERF" = true ]; then
    echo ""
    echo "使用perf进行性能分析..."
    
    # 创建性能分析目录
    mkdir -p results/perf_results
    
    # 检查perf是否支持所有事件
    DETAILED_EVENTS="cycles,instructions,cache-references,cache-misses,branch-misses"
    CACHE_EVENTS=""
    
    # 检查不同的缓存事件是否可用
    if perf list 2>&1 | grep -q "L1-dcache-loads"; then
        CACHE_EVENTS="$CACHE_EVENTS,L1-dcache-loads,L1-dcache-load-misses"
    fi
    
    if perf list 2>&1 | grep -q "LLC-loads"; then
        CACHE_EVENTS="$CACHE_EVENTS,LLC-loads,LLC-load-misses"
    fi
    
    if perf list 2>&1 | grep -q "dTLB-loads"; then
        CACHE_EVENTS="$CACHE_EVENTS,dTLB-loads,dTLB-load-misses"
    fi
    
    # 基本系统事件，应在大多数系统上工作
    SYS_EVENTS="task-clock,context-switches,cpu-migrations,page-faults"
    
    # 合并事件列表
    ALL_EVENTS="$DETAILED_EVENTS$CACHE_EVENTS,$SYS_EVENTS"
    ALG_EVENTS="$DETAILED_EVENTS$CACHE_EVENTS,task-clock"
    
    # 基本性能统计
    echo "收集基本性能统计..."
    perf stat -o results/perf_results/matrix_basic_stat.txt \
              ./task1_matrix_test random 3000 quick
    
    # 详细的CPU周期和缓存统计
    echo "收集详细的CPU性能指标..."
    perf stat -e $ALL_EVENTS \
              -o results/perf_results/matrix_detailed_stats.txt \
              ./task1_matrix_test random 3000 quick
    
    # IPC（每周期指令数）计算
    echo "计算IPC（每周期指令数）..."
    perf stat -e cycles,instructions \
              -o results/perf_results/matrix_ipc.txt \
              ./task1_matrix_test random 3000 quick
    
    # 热点函数分析
    echo "进行热点函数分析..."
    perf record -g -o results/perf_results/matrix_hotspots.data \
                ./task1_matrix_test random 3000 quick
    perf report -i results/perf_results/matrix_hotspots.data --stdio > results/perf_results/matrix_hotspots_report.txt
    
    # 调用图生成
    echo "生成调用图..."
    perf record -g -o results/perf_results/matrix_callgraph.data \
                ./task1_matrix_test random 2000 quick
    
    # 尝试生成火焰图（如果可能）
    if command -v perl &> /dev/null; then
        # 检查是否存在FlameGraph工具
        FLAMEGRAPH_FOUND=false
        for FG_PATH in "./FlameGraph" "../FlameGraph" "/opt/FlameGraph" "$HOME/FlameGraph"; do
            if [ -d "$FG_PATH" ] && [ -f "$FG_PATH/stackcollapse-perf.pl" ] && [ -f "$FG_PATH/flamegraph.pl" ]; then
                echo "发现FlameGraph工具在: $FG_PATH"
                perf script -i results/perf_results/matrix_callgraph.data | \
                "$FG_PATH/stackcollapse-perf.pl" | \
                "$FG_PATH/flamegraph.pl" > results/perf_results/matrix_flamegraph.svg
                
                if [ $? -eq 0 ]; then
                    echo "火焰图已保存到 results/perf_results/matrix_flamegraph.svg"
                    FLAMEGRAPH_FOUND=true
                    break
                fi
            fi
        done
        
        if [ "$FLAMEGRAPH_FOUND" = false ]; then
            echo "未找到FlameGraph工具，跳过火焰图生成"
            echo "如需生成火焰图，请从 https://github.com/brendangregg/FlameGraph 获取工具"
        fi
    else
        echo "未安装perl，无法生成火焰图"
    fi
    
    # 不同算法的详细分析
    echo "比较不同算法的性能特性..."
    
    # 朴素算法
    echo "分析朴素算法性能..."
    perf stat -e $ALG_EVENTS \
              -o results/perf_results/naive_stats.txt \
              ./task1_matrix_test random 2000 naive_only
    
    # 缓存优化算法
    echo "分析缓存优化算法性能..."
    perf stat -e $ALG_EVENTS \
              -o results/perf_results/cache_opt_stats.txt \
              ./task1_matrix_test random 2000 cache_opt_only
    
    # OpenMP并行算法
    echo "分析OpenMP并行算法性能..."
    perf stat -e $ALG_EVENTS,context-switches,cpu-migrations \
              -o results/perf_results/openmp_stats.txt \
              ./task1_matrix_test random 2000 openmp_only
    
    # 分块算法
    echo "分析分块算法性能..."
    perf stat -e $ALG_EVENTS \
              -o results/perf_results/blocked_stats.txt \
              ./task1_matrix_test random 2000 blocked_only
    
    # 进行更大规模的性能分析测试
    echo "进行大规模矩阵测试性能分析..."
    perf stat -e cycles,instructions,cache-references,cache-misses \
              -o results/perf_results/large_matrix_test.txt \
              ./task1_matrix_test random 4000 quick
    
    echo "性能分析完成，结果保存在 results/perf_results/ 目录中"
    
    # 尝试生成性能报告摘要
    echo "生成性能报告摘要..."
    {
        echo "========== 矩阵乘法性能分析摘要 =========="
        echo "日期: $(date)"
        echo "系统: $(uname -a)"
        echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)"
        echo "核心数: $CORE_COUNT"
        echo ""
        echo "========== 算法性能比较 =========="
        echo "朴素算法:"
        grep -A5 "Performance counter stats" results/perf_results/naive_stats.txt 2>/dev/null || echo "数据不可用"
        echo ""
        echo "缓存优化算法:"
        grep -A5 "Performance counter stats" results/perf_results/cache_opt_stats.txt 2>/dev/null || echo "数据不可用"
        echo ""
        echo "OpenMP并行算法:"
        grep -A5 "Performance counter stats" results/perf_results/openmp_stats.txt 2>/dev/null || echo "数据不可用"
        echo ""
        echo "分块算法:"
        grep -A5 "Performance counter stats" results/perf_results/blocked_stats.txt 2>/dev/null || echo "数据不可用"
        echo ""
        echo "完整结果请查看 results/perf_results/ 目录中的各个文件"
    } > results/perf_results/performance_summary.txt
fi

# 生成可视化图表
echo ""
echo "生成性能分析图表..."
python3 plot_matrix_performance.py

echo ""
echo "测试完成。结果已保存在 results/ 目录中"
if [ "$HAS_PERF" = true ]; then
    echo "性能分析结果已保存在 results/perf_results/ 目录中"
    echo "性能摘要可在 results/perf_results/performance_summary.txt 中查看"
fi