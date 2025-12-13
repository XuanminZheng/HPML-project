#!/bin/bash

CONFIG=$1

case $CONFIG in
    "no_prefix")
        echo "Starting WITHOUT prefix caching..."
        vllm serve openbmb/MiniCPM-V-4 \
            --dtype auto \
            --max-model-len 4096 \
            --trust-remote-code \
            --gpu_memory_utilization 0.9 \
            --no-enable-prefix-caching \
            --port 8000 2>&1 | tee logs/no_prefix.log
        ;;
    
    "prefix_caching")
        echo "Starting WITH prefix caching (explicit)..."
        vllm serve openbmb/MiniCPM-V-4 \
            --dtype auto \
            --max-model-len 4096 \
            --trust-remote-code \
            --gpu_memory_utilization 0.9 \
            --enable-prefix-caching \
            --port 8000 2>&1 | tee logs/prefix_caching.log
        ;;
    
    "chunked_prefill")
        echo "Starting with chunked prefill..."
        vllm serve openbmb/MiniCPM-V-4 \
            --dtype auto \
            --max-model-len 4096 \
            --trust-remote-code \
            --gpu_memory_utilization 0.9 \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            --max-num-batched-tokens 2048 \
            --port 8000 2>&1 | tee logs/chunked_prefill.log
        ;;
    
    *)
        echo "Usage: ./start_kv_configs.sh [no_prefix|prefix_caching|chunked_prefill]"
        exit 1
        ;;
esac