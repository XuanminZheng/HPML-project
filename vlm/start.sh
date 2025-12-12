#!/bin/bash

vllm serve openbmb/MiniCPM-V-4 \
    --dtype auto \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu_memory_utilization 0.9 \
    --port 8000 > vllm.log 2>&1 &

echo "vLLM 已在后台启动，PID: $!"
echo "查看日志: tail -f vllm.log"