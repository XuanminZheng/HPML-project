#!/bin/bash

vllm serve openbmb/MiniCPM-V-4 \
    --dtype auto \
    --quantization bitsandbytes \
    --max-model-len 4096 \
    --trust-remote-code \
    --port 8001 > vllm_int4.log 2>&1 &

echo "[INT4] vLLM launched，PID: $!"
