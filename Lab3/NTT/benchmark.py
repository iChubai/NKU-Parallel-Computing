import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys

def check_files():
    required_files = [
        'pthread/main_ptread_v1.cc',
        'pthread/main_ptread_v2.cc',
        'pthread/main_ptread_v3.cc'
    ]
    for file in required_files:
        if not os.path.exists(file):
            print(f"错误：找不到文件 {file}")
            return False
    return True

def compile_code(version, threads):
    try:
        # 修改代码中的线程数
        with open(f'pthread/main_ptread_v{version}.cc', 'r') as f:
            content = f.read()
        
        # 替换线程数定义
        content = re.sub(r'#define V\d_THREADS \d+', f'#define V{version}_THREADS {threads}', content)
        
        # 写入临时文件
        temp_file = f'temp_v{version}.cc'
        with open(temp_file, 'w') as f:
            f.write(content)
        
        # 编译
        result = subprocess.run(['g++', '-O3', '-pthread', temp_file, '-o', f'bench_v{version}'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"编译错误：\n{result.stderr}")
            return False
            
        os.remove(temp_file)
        return True
    except Exception as e:
        print(f"编译过程出错：{str(e)}")
        return False

def run_benchmark(version, threads):
    try:
        if not compile_code(version, threads):
            return None
            
        results = []
        # 运行5次取平均值
        for i in range(5):
            print(f"    运行第 {i+1} 次...")
            output = subprocess.check_output(['./bench_v' + str(version)], text=True)
            # 提取延迟数据
            match = re.search(r': ([\d.]+) \(us\)', output)
            if not match:
                print(f"无法从输出中提取延迟数据：\n{output}")
                return None
            latency = float(match.group(1))
            results.append(latency)
        
        return np.mean(results)
    except Exception as e:
        print(f"运行基准测试时出错：{str(e)}")
        return None

def main():
    if not check_files():
        sys.exit(1)
        
    thread_counts = [1, 2, 4, 8]
    versions = [1, 2, 3]
    results = {v: [] for v in versions}
    
    # 运行基准测试
    for version in versions:
        print(f"\n测试版本 {version}...")
        for threads in thread_counts:
            print(f"  线程数: {threads}")
            latency = run_benchmark(version, threads)
            if latency is None:
                print(f"  测试失败，跳过当前配置")
                continue
            results[version].append(latency)
            print(f"  平均延迟: {latency:.2f} us")
    
    # 检查是否有有效数据
    if not any(results.values()):
        print("没有收集到有效数据，无法生成图表")
        sys.exit(1)
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    for version in versions:
        if results[version]:  # 只绘制有数据的版本
            plt.plot(thread_counts[:len(results[version])], results[version], 
                    marker='o', label=f'Version {version}')
    
    plt.xlabel('线程数')
    plt.ylabel('平均延迟 (us)')
    plt.title('不同线程数对NTT性能的影响')
    plt.grid(True)
    plt.legend()
    
    # 保存图表
    plt.savefig('benchmark_results.png')
    plt.close()
    
    # 保存数据
    with open('benchmark_results.txt', 'w') as f:
        f.write('Threads\tV1\tV2\tV3\n')
        for i, threads in enumerate(thread_counts):
            v1 = f"{results[1][i]:.2f}" if i < len(results[1]) else "N/A"
            v2 = f"{results[2][i]:.2f}" if i < len(results[2]) else "N/A"
            v3 = f"{results[3][i]:.2f}" if i < len(results[3]) else "N/A"
            f.write(f'{threads}\t{v1}\t{v2}\t{v3}\n')
    
    print("\n基准测试完成！")
    print("结果已保存到 benchmark_results.png 和 benchmark_results.txt")

if __name__ == '__main__':
    main() 