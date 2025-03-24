#!/bin/bash

# 显示 CPU 架构信息
echo "CPU Architecture Information:"
lscpu
echo ""

# 显示 /proc/cpuinfo 内容
echo "Detailed CPU Info from /proc/cpuinfo:"
cat /proc/cpuinfo
echo ""

# 显示 CPU 频率信息
echo "CPU Frequency Information:"
cat /proc/cpuinfo | grep "MHz"
echo ""

# 显示 CPU 缓存信息
echo "CPU Cache Information:"
lscpu | grep "Cache"
echo ""

# 显示 NUMA 信息
echo "NUMA Information:"
lscpu | grep "NUMA"
echo ""

# 显示 CPU 漏洞信息
echo "CPU Vulnerabilities:"
cat /proc/cpuinfo | grep "Vulnerabilities"
echo ""

# 完成
echo "CPU information displayed successfully."
