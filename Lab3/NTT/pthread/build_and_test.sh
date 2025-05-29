#!/bin/bash

# =============================================================================
# 文件名: build_and_test.sh
# 描述: 编译和测试NTT高级优化算法
# 用法: ./build_and_test.sh
# =============================================================================

set -e  # 遇到错误时退出

echo "=========================================="
echo "NTT高级优化算法编译和测试脚本"
echo "=========================================="

# 检查编译器
if ! command -v g++ &> /dev/null; then
    echo "错误: 需要g++编译器"
    exit 1
fi

# 检查系统架构
ARCH=$(uname -m)
echo "检测到系统架构: $ARCH"

# 根据架构设置编译选项
if [[ "$ARCH" == "x86_64" ]]; then
    echo "使用x86_64优化编译选项"
    ARCH_FLAGS="-march=native -mavx2 -mfma"
    SIMD_FLAGS="-D__AVX2__"
elif [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]]; then
    echo "使用ARM64优化编译选项"
    ARCH_FLAGS="-march=native -mcpu=native"
    SIMD_FLAGS="-D__ARM_NEON"
else
    echo "使用通用编译选项"
    ARCH_FLAGS="-march=native"
    SIMD_FLAGS=""
fi

# 检查编译器支持
echo "检查编译器支持..."
g++ --version

# 基础编译选项
BASE_FLAGS="-std=c++17 -O2 -pthread -Wall -Wextra"
OPENMP_FLAGS="-fopenmp"

# 创建构建目录
mkdir -p build
cd build

echo ""
echo "1. 编译Cache-Oblivious NTT算法..."

# 尝试不同的编译选项组合
COMPILE_SUCCESS=0

# 尝试1: 完整优化
if [ $COMPILE_SUCCESS -eq 0 ]; then
    echo "  尝试完整优化编译..."
    if g++ $BASE_FLAGS $ARCH_FLAGS $SIMD_FLAGS $OPENMP_FLAGS \
        -DCACHE_OBLIVIOUS_MAIN \
        -o cache_oblivious_ntt ../cache_oblivious_ntt.cc 2>/dev/null; then
        echo "✓ Cache-Oblivious NTT编译成功 (完整优化)"
        COMPILE_SUCCESS=1
    fi
fi

# 尝试2: 无SIMD优化
if [ $COMPILE_SUCCESS -eq 0 ]; then
    echo "  尝试无SIMD优化编译..."
    if g++ $BASE_FLAGS $OPENMP_FLAGS \
        -DCACHE_OBLIVIOUS_MAIN \
        -o cache_oblivious_ntt ../cache_oblivious_ntt.cc 2>/dev/null; then
        echo "✓ Cache-Oblivious NTT编译成功 (无SIMD)"
        COMPILE_SUCCESS=1
    fi
fi

# 尝试3: 最小化选项
if [ $COMPILE_SUCCESS -eq 0 ]; then
    echo "  尝试最小化选项编译..."
    if g++ -std=c++17 -O2 -pthread \
        -DCACHE_OBLIVIOUS_MAIN \
        -o cache_oblivious_ntt ../cache_oblivious_ntt.cc 2>/dev/null; then
        echo "✓ Cache-Oblivious NTT编译成功 (最小化)"
        COMPILE_SUCCESS=1
    fi
fi

if [ $COMPILE_SUCCESS -eq 0 ]; then
    echo "✗ Cache-Oblivious NTT编译失败"
    echo "  尝试显示编译错误:"
    g++ -std=c++17 -O2 -pthread \
        -DCACHE_OBLIVIOUS_MAIN \
        -o cache_oblivious_ntt ../cache_oblivious_ntt.cc || true
fi

echo ""
echo "2. 编译Work-Stealing NTT算法..."
if g++ $BASE_FLAGS \
    -DWORK_STEALING_MAIN \
    -o work_stealing_ntt ../work_stealing_ntt.cc 2>/dev/null; then
    echo "✓ Work-Stealing NTT编译成功"
else
    echo "✗ Work-Stealing NTT编译失败"
    echo "  尝试显示编译错误:"
    g++ $BASE_FLAGS \
        -DWORK_STEALING_MAIN \
        -o work_stealing_ntt ../work_stealing_ntt.cc || true
fi

echo ""
echo "3. 编译综合性能测试..."
if g++ $BASE_FLAGS \
    -o ntt_benchmark ../advanced_ntt_benchmark.cc 2>/dev/null; then
    echo "✓ 综合性能测试编译成功"
else
    echo "✗ 综合性能测试编译失败"
    echo "  尝试显示编译错误:"
    g++ $BASE_FLAGS \
        -o ntt_benchmark ../advanced_ntt_benchmark.cc || true
fi

echo ""
echo "4. 编译现有的pthread实现进行对比..."

# 编译现有的实现
for file in ../main_pthread*.cc ../crt_ptread.cc; do
    if [ -f "$file" ]; then
        basename=$(basename "$file" .cc)
        echo "编译 $basename..."
        if g++ $BASE_FLAGS \
            -o "$basename" "$file" 2>/dev/null; then
            echo "✓ $basename 编译成功"
        else
            echo "⚠ $basename 编译失败"
        fi
    fi
done

echo ""
echo "=========================================="
echo "开始性能测试"
echo "=========================================="

# 运行测试
echo ""
echo "1. 测试Cache-Oblivious NTT (如果编译成功)..."
if [ -f "cache_oblivious_ntt" ]; then
    echo "运行Cache-Oblivious NTT演示:"
    timeout 30s ./cache_oblivious_ntt || echo "  (测试超时或失败)"
else
    echo "  跳过 - 未编译成功"
fi

echo ""
echo "2. 测试Work-Stealing NTT (如果编译成功)..."
if [ -f "work_stealing_ntt" ]; then
    echo "运行Work-Stealing NTT演示:"
    timeout 30s ./work_stealing_ntt || echo "  (测试超时或失败)"
else
    echo "  跳过 - 未编译成功"
fi

echo ""
echo "3. 运行综合性能测试 (如果编译成功)..."
if [ -f "ntt_benchmark" ]; then
    echo "运行综合性能基准测试:"
    timeout 60s ./ntt_benchmark || echo "  (测试超时或失败)"
else
    echo "  跳过 - 未编译成功"
fi

echo ""
echo "=========================================="
echo "系统信息和性能对比"
echo "=========================================="

echo "系统信息:"
echo "  架构: $ARCH"
echo "  CPU信息:"
cat /proc/cpuinfo | grep "model name" | head -1 || echo "  无法获取CPU信息"
echo "  CPU核心数: $(nproc)"
echo "  内存信息:"
free -h | head -2 || echo "  无法获取内存信息"

# 简单的性能对比
echo ""
echo "运行小规模性能对比..."

test_sizes=(256 1024)
for size in "${test_sizes[@]}"; do
    echo ""
    echo "测试规模 N=$size:"
    
    # 如果有现有实现，运行它们
    for exe in main_pthread main_pthread_v* main_radix-4_pthread; do
        if [ -f "$exe" ]; then
            echo -n "  $exe: "
            timeout 10s ./"$exe" 2>/dev/null | tail -1 || echo "超时/失败"
        fi
    done
done

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="

echo ""
echo "生成的文件:"
ls -la *.csv *.out *.png 2>/dev/null || echo "无输出文件"

echo ""
echo "编译的可执行文件:"
ls -la cache_oblivious_ntt work_stealing_ntt ntt_benchmark main_* 2>/dev/null || echo "无可执行文件"