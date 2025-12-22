# HPML Project 

An open-source project for optimizing and evaluating the inference performance and accuracy of Large Language Models (LLM) and Vision Language Models (VLM). The project includes two main modules: LLM and VLM, supporting multiple optimization strategies including quantization (INT4), operator fusion optimization.

## 📁 Project Structure

```
HPML-project/
├── llm/                           # Large Language Model Module
│   └── main.py                    # LLM inference example (TinyLlama)
│
├── vlm/                           # Vision Language Model Module
│   ├── main.py                    # VLM inference example (MiniCPM-V-4)
│   ├── api.py                     # Performance analysis and benchmarking script
│   ├── fusion.py                  # Fused operator implementation (LayerNorm, Attention, FFN)
│   ├── test_acc.py                # Model accuracy evaluation script (MME dataset)
│   ├── run.sh                     # Batch testing script
│   ├── run_all_perf.sh            # Full performance testing script
│   ├── start.sh / start_int4.sh   # Service startup scripts
│   ├── remove_cuda.sh             # CUDA environment cleanup script
│   ├── img/                       # Test image directory
│   └── [perf_*.json/mme_*.json]   # Performance and accuracy test results
```

## 🎯 Core Features

### 1. **LLM Module**

- Efficient inference based on vLLM framework
- Supports TinyLlama-1.1B-Chat model
- Both streaming and non-streaming generation modes

### 2. **VLM Module**

- Vision language model inference based on vLLM framework
- Supports MiniCPM-V-4 model (multimodal)
- Multiple optimization configurations:
  - **baseline**: Standard model
  - **int4**: INT4 quantized version
  - **int8**: INT8 quantized version (in development)
  - **activation**: Activation function optimization
  - **fusion**: Operator fusion optimization

### 3. **Performance Analysis**

Provides detailed performance metrics including:

- **TTFT** (Time to First Token): First token latency
- **Prefill/Decode Speed**: Input processing and output generation speed
- **End-to-End Latency**: Total response time
- **Throughput**: Requests/s and Tokens/s
- **GPU Memory Usage**: Memory consumption and utilization

### 4. **Accuracy Evaluation**

- Based on MME (Multimodal Multitask Evaluation) dataset
- Supports accuracy comparison across multiple model configurations
- Automatic result saving

### 5. **Operator Fusion Optimization**

Implements multiple fusion strategies to improve computational efficiency:

- **FusedAttentionBlock**: Fused LayerNorm + Attention
- **FusedFFNBlock**: Fused LayerNorm + FFN + Activation
- **FusedTransformerBlock**: Complete Transformer block fusion
- Provides performance benchmark comparisons

## 🚀 Quick Start

### Environment Dependencies

```bash
pip install vllm
pip install torch torchvision torchaudio
pip install pillow
pip install openai
pip install datasets
```

### 1. Basic Inference Examples

#### LLM Inference

```bash
cd llm
python main.py
```

#### VLM Inference

```bash
cd vlm
python main.py
```

### 2. Start vLLM Service

**Baseline Model**

```bash
bash start.sh
```

**INT4 Quantized Version**

```bash
bash start_int4.sh
```

### 3. Performance Benchmarking

```bash
# Make sure vLLM API service is running
python api.py --config baseline --port 8000
python api.py --config int4 --port 8001
```

Output Example:

```
============================================================
PERFORMANCE METRICS REPORT
============================================================
⏱️  Time to First Token (TTFT):
  Average:    234.56 ms
  Min:        210.32 ms
  Max:        256.78 ms
  Std Dev:    15.43 ms

📊 Decode Speed:
  Average:    45.32 tokens/s
  Min:        42.15 tokens/s
  Max:        48.76 tokens/s

💾 GPU Memory:
  Used:        24500 MiB
  Total:       40960 MiB
  Utilization: 59.8%
```

### 4. Accuracy Evaluation

```bash
# Test specific configuration
python test_acc.py --config baseline --num-samples 100

# Test all configurations
python test_acc.py --num-samples 50

# Batch testing
bash run.sh 50  # Test 50 samples
```

Results will be saved to `mme_accuracy_results.json`

### 5. Operator Fusion Benchmark

```bash
cd vlm
python fusion.py
```

Output Example:

```
Using device: cuda
Fused:    234.56ms
Standard: 312.45ms
Speedup:  1.33x
```
```
python batch_strategy.py
```

## 🔧 Main Script Documentation

### `api.py` - Performance Analysis Script

```bash
python api.py --config [baseline|int4] --port [PORT]
```

**Parameters:**

- `--config`: Model configuration (baseline/int4)
- `--port`: vLLM service port (default: 8000)

**Output:**

- Console output with detailed performance metrics
- JSON file with complete performance data

### `test_acc.py` - Accuracy Testing Script

```bash
python test_acc.py [--config CONFIG] [--num-samples N]
```

**Parameters:**

- `--config`: Test configuration (baseline/int4/fusion)
- `--num-samples`: Number of test samples (default: all)

**Output:**

- Console display of test progress and results
- JSON file with test results

### `fusion.py` - Operator Fusion Benchmark

Implements performance comparison between fused and standard Transformer blocks.

**Key Classes:**

- `FusedAttentionBlock`: Fused attention block
- `FusedFFNBlock`: Fused feed-forward network block
- `FusedTransformerBlock`: Complete fused Transformer block
- `StandardTransformerBlock`: Standard implementation (for comparison)

## 🛠️ Custom Configuration

### Modify Model Parameters

In `main.py`:

```python
llm = LLM(
    model='openbmb/MiniCPM-V-4',
    trust_remote_code=True,
    gpu_memory_utilization=0.9,      # Adjust GPU memory utilization
    max_model_len=4096                # Adjust maximum sequence length
)
```

### Modify Inference Parameters

```python
params = SamplingParams(
    max_tokens=512,                   # Adjust max generation length
    temperature=0.7                   # Adjust temperature parameter
)
```

## 🔍 Troubleshooting

### Out of GPU Memory

```bash
# 1. Lower gpu_memory_utilization
# 2. Use quantized version (INT4)
# 3. Reduce max_model_len
```

### CUDA Issues

```bash
bash vlm/remove_cuda.sh  # Clean CUDA cache
```

### Slow Model Loading

```bash
# Pre-download models
python -c "from vllm import LLM; LLM('openbmb/MiniCPM-V-4')"
```

## 📚 Related Resources

- [vLLM Official Documentation](https://docs.vllm.ai/)
- [MiniCPM-V Project](https://github.com/OpenBMB/MiniCPM-V)
- [MME Dataset](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)

## 📝 Paper References

If you use this project, please refer to these papers:

- vLLM: Efficient Memory Management for Large Language Model Serving
- MiniCPM: Efficient End-side Inference for Multimodal Large Language Models
- Operator Fusion for Neural Network Optimization

## 📄 License

[To be supplemented]

## 👥 Contributors

* Zhongyun Liu
* Zhengbin Lu
* Xuanmin Zheng

## 🤝 Contributing Guide

Issues and suggestions for improvements are welcome!

### How to Contribute

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/xxx`)
3. Commit your changes (`git commit -m 'Add xxx'`)
4. Push to the branch (`git push origin feature/xxx`)
5. Create a Pull Request

## Frequently Asked Questions (FAQ)

**Q: How to test other models?**

A: Modify the model name in `main.py`:

```python
llm = LLM('meta-llama/Llama-2-7b-chat-hf', ...)
```

**Q: How to export performance data for analysis?**

A: Performance data is automatically saved to `perf_*.json`. Use any JSON reader or Python:

```python
import json
with open('perf_baseline.json') as f:
    data = json.load(f)
```

**Q: Does it support multi-GPU inference?**

A: Current version doesn't have explicit support, but can be implemented by modifying vLLM configuration.

**Q: How to use offline?**

A: Pre-download model weights and specify the local path:

```python
llm = LLM('/path/to/local/model')
```

---

**Last Updated:** 2025-01-01
