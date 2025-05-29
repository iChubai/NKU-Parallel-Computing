#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 切换到脚本所在目录
cd "$SCRIPT_DIR"

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误：未找到Python3，请先安装Python3"
    exit 1
fi

# 检查必要的目录是否存在
if [ ! -d "pthread" ]; then
    echo "错误：找不到pthread目录"
    echo "当前目录: $(pwd)"
    echo "目录内容:"
    ls -la
    exit 1
fi

# 创建输出目录
mkdir -p files

# 安装必要的Python包
echo "安装必要的Python包..."
python3 -m pip install --user matplotlib numpy

# 运行基准测试
echo "开始运行基准测试..."
python3 benchmark.py

# 检查是否成功生成结果文件
if [ -f "benchmark_results.png" ] && [ -f "benchmark_results.txt" ]; then
    echo "基准测试完成！"
    echo "结果已保存到："
    echo "- benchmark_results.png（性能图表）"
    echo "- benchmark_results.txt（详细数据）"
else
    echo "错误：基准测试可能未成功完成"
    exit 1
fi 