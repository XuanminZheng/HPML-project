#!/usr/bin/env python3
"""
MME Accuracy Test for MiniCPM-V-4
Tests: baseline, int4, int8, activation, fusion
"""

import json
import os
import sys
import torch
import argparse
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import io
import time
import base64

# 根据配置选择模型加载方式
def load_model(config, port=None):
    """加载模型，根据配置选择不同的模型"""
    from openai import OpenAI

    # 所有API配置（包括activation）的端口映射
    port_map = {
        "baseline": 8000,
        "int4": 8001,
    }

    if config == "fusion":
        # 融合模型 - 直接使用vLLM而不是API
        from vllm import LLM, SamplingParams
        print("  Loading fusion model via vLLM...")
        llm = LLM(
            model='openbmb/MiniCPM-V-4',
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=4096
        )
        return llm, SamplingParams(max_tokens=512, temperature=0.7)
    else:
        # 其他所有配置（baseline, int4, int8, activation）都通过API调用
        if port is None:
            port = port_map.get(config, 8000)

        api_base = f"http://localhost:{port}/v1"
        print(f"  Connecting to API at {api_base}...")
        client = OpenAI(base_url=api_base, api_key="token")
        return client, None


def get_mme_dataset():
    """加载MME数据集"""
    try:
        dataset = load_dataset("lmms-lab/MME", split="test")
        print(f"✓ Loaded MME dataset: {len(dataset)} samples")
        return dataset
    except Exception as e:
        print(f"✗ Failed to load MME dataset: {e}")
        print("Trying alternative loading method...")
        return None


def extract_answer(text):
    """从模型输出中提取答案（Yes/No）"""
    text = text.strip().upper()
    if "YES" in text:
        return "yes"
    elif "NO" in text:
        return "no"
    else:
        # 如果不确定，返回模型输出的第一个字符
        return text[0].lower() if text else "unknown"


def infer_with_api(client, image_data, question):
    """通过API进行推理"""
    import base64

    if isinstance(image_data, Image.Image):
        # PIL Image转base64
        buffered = io.BytesIO()
        image_data.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
    else:
        img_base64 = image_data

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
        ]
    }]

    response = client.chat.completions.create(
        model="openbmb/MiniCPM-V-4",
        messages=messages,
        max_tokens=10,
        temperature=0.0
    )

    return response.choices[0].message.content



def infer_with_vllm(llm, sampling_params, image, question):
    """通过vLLM进行推理（使用 OpenAI 格式的 image_url）"""


    if isinstance(image, Image.Image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
    else:
        # 如果已经是 base64 字符串
        img_base64 = image

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
        ]
    }]

    outputs = llm.chat(messages, sampling_params=sampling_params)
    return outputs[0].outputs[0].text.strip()


def test_accuracy(config, num_samples=None):
    """测试accuracy"""
    print(f"\n{'='*60}")
    print(f"Testing Config: {config}")
    print(f"{'='*60}")

    # 加载模型
    print("Loading model...")
    try:
        if config == "fusion":
            model, params = load_model(config)
            use_api = False
        else:
            model, params = load_model(config)
            use_api = True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None

    # 加载数据集
    dataset = get_mme_dataset()
    if dataset is None:
        print("✗ Cannot proceed without dataset")
        return None

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    # 开始测试
    correct = 0
    total = 0
    errors = []
    start_time = time.time()

    for idx, sample in enumerate(dataset):
        try:
            # 获取图片和问题
            if "image" in sample:
                image = sample["image"]
                if isinstance(image, dict) and "bytes" in image:
                    image = Image.open(io.BytesIO(image["bytes"]))
            else:
                continue

            question = sample.get("question", "")
            gt_answer = sample.get("answer", "").lower()

            if not question or not gt_answer:
                continue

            # 推理
            if use_api:
                pred_text = infer_with_api(model, image, question)
            else:
                pred_text = infer_with_vllm(model, params, image, question)

            pred_answer = extract_answer(pred_text)

            # 判断是否正确
            is_correct = (pred_answer == gt_answer or
                          pred_text.lower().strip() == gt_answer)

            if is_correct:
                correct += 1
            else:
                if len(errors) < 5:  # 只记录前5个错误
                    errors.append({
                        "question": question,
                        "gt": gt_answer,
                        "pred": pred_answer,
                        "raw": pred_text[:50]
                    })

            total += 1

            if (idx + 1) % 10 == 0:
                acc = correct / total * 100 if total > 0 else 0
                print(f"  [{idx+1}/{len(dataset)}] Current Accuracy: {acc:.1f}%")

        except Exception as e:
            print(f"  ✗ Error on sample {idx}: {str(e)[:50]}")
            continue

    elapsed = time.time() - start_time

    # 计算结果
    if total == 0:
        print("✗ No valid samples tested")
        return None

    accuracy = correct / total * 100

    # 打印结果
    print(f"\n{'='*60}")
    print(f"RESULTS for {config}")
    print(f"{'='*60}")
    print(f"Accuracy:    {accuracy:.2f}% ({correct}/{total})")
    print(f"Total Time:  {elapsed:.1f}s")
    print(f"Time/Sample: {elapsed/total:.2f}s")

    if errors:
        print(f"\nSample Errors:")
        for err in errors:
            print(f"  Q: {err['question'][:40]}...")
            print(f"  GT: {err['gt']}, Pred: {err['pred']}, Raw: {err['raw']}")

    result = {
        "config": config,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "time_total": elapsed,
        "time_per_sample": elapsed / total
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, choices=["baseline", "int4", "fusion"],
                        help="Specific config to test (default: all 5)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to test (default: all)")
    args = parser.parse_args()

    # 默认测试所有5个配置
    if args.config:
        configs = [args.config]
    else:
        configs = ["baseline", "int4", "fusion"]

    print(f"\n Testing {len(configs)} config(s): {', '.join(configs)}\n")

    all_results = []
    for config in configs:
        result = test_accuracy(config, num_samples=args.num_samples)
        if result:
            all_results.append(result)

    # 汇总结果
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for r in all_results:
            print(f"{r['config']:12} | Accuracy: {r['accuracy']:6.2f}% | "
                  f"Time: {r['time_total']:6.1f}s | "
                  f"Avg: {r['time_per_sample']:.2f}s/sample")

        # 保存结果（支持追加）
        results_file = "mme_accuracy_results.json"
        existing_results = []

        # 如果文件已存在，读取现有结果
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    existing_results = json.load(f)
            except:
                existing_results = []

        # 移除重复的配置结果（同一配置只保留最新的）
        existing_results = [r for r in existing_results if r['config'] not in [x['config'] for x in all_results]]

        # 合并结果
        all_results = existing_results + all_results

        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()