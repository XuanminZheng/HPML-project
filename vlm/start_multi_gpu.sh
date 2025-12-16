#!/bin/bash

export HF_HOME=/var/hf_cache

# 停止现有服务
pkill -f "vllm serve" 2>/dev/null
sleep 2

# GPU 0: 端口 8000
CUDA_VISIBLE_DEVICES=0 vllm serve openbmb/MiniCPM-V-4 \
    --dtype auto \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu_memory_utilization 0.9 \
    --enable-prefix-caching \
    --port 8000 > vllm_gpu0.log 2>&1 &

echo "GPU 0 已在后台启动，PID: $!, 端口: 8000"

# GPU 1: 端口 8001
CUDA_VISIBLE_DEVICES=1 vllm serve openbmb/MiniCPM-V-4 \
    --dtype auto \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu_memory_utilization 0.9 \
    --enable-prefix-caching \
    --port 8001 > vllm_gpu1.log 2>&1 &

echo "GPU 1 已在后台启动，PID: $!, 端口: 8001"

echo ""
echo "查看日志:"
echo "  tail -f vllm_gpu0.log"
echo "  tail -f vllm_gpu1.log"