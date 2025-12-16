#!/bin/bash

NUM_SAMPLES=${1:-50}
RESULTS_FILE="mme_accuracy_results.json"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "------    MME Accuracy Test   ------"
echo "------Configs: baseline, int4, fusion    ------"
echo -e "${NC}"
echo "num of sample: $NUM_SAMPLES"
echo ""

# 清空之前的结果文件
rm -f $RESULTS_FILE

# 配置列表
declare -a CONFIGS=("baseline" "int4" "fusion")
declare -a PORTS=("8000" "8001" "0")

# 验证服务是否在线（fusion除外，因为用SDK直接推理）
echo -e "${YELLOW}checking services...${NC}"
for i in "${!CONFIGS[@]}"; do
    if [ "${CONFIGS[$i]}" != "fusion" ]; then
        if curl -s http://localhost:${PORTS[$i]}/v1/models > /dev/null 2>&1; then
            echo -e "${GREEN}${CONFIGS[$i]} (port ${PORTS[$i]}) is running${NC}"
        else
            echo -e "${RED}❌ ${CONFIGS[$i]} (port ${PORTS[$i]}) is NOT running${NC}"
        fi
    else
        echo -e "${GREEN}fusion (SDK inference) ready${NC}"
    fi
done
echo ""

# 按顺序测试每个配置（跳过启动步骤）
for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    PORT="${PORTS[$i]}"

    echo -e "${BLUE}------ test ${CONFIG} (${i+1}/${#CONFIGS[@]})------${NC}"

    if [ "$CONFIG" = "fusion" ]; then
        echo -e "${YELLOW}run test with SDK (fusion)...${NC}"
    else
        echo -e "${YELLOW}run test on port $PORT...${NC}"
    fi

    # 直接运行测试
    if [ "$NUM_SAMPLES" -gt 0 ]; then
        python test_acc.py --config "$CONFIG" --num-samples "$NUM_SAMPLES"
    else
        python test_acc.py --config "$CONFIG"
    fi

    echo ""
done

# 打印最终结果汇总
echo ""
echo -e "${BLUE} ------                  results                ------${NC}"

if [ -f "$RESULTS_FILE" ]; then
    python << 'EOF'
import json

with open("mme_accuracy_results.json", "r") as f:
    results = json.load(f)

print("\n配置         | Accuracy  | 总耗时 | 单样本耗时")
print("-" * 55)
for r in results:
    config = r['config'].ljust(12)
    acc = f"{r['accuracy']:.2f}%".rjust(8)
    total_time = f"{r['time_total']:.1f}s".rjust(7)
    per_sample = f"{r['time_per_sample']:.2f}s".rjust(10)
    print(f"{config} | {acc} | {total_time} | {per_sample}")

# 找出最高和最低精度
accs = [r['accuracy'] for r in results]
best_idx = accs.index(max(accs))
worst_idx = accs.index(min(accs))
print("\n" + "-" * 55)
print(f"最高精度: {results[best_idx]['config']} ({max(accs):.2f}%)")
print(f"最低精度: {results[worst_idx]['config']} ({min(accs):.2f}%)")
print(f"平均精度: {sum(accs)/len(accs):.2f}%")
EOF

    echo ""
    echo -e "${GREEN}test finished！${NC}"
    echo "save as: $RESULTS_FILE"
else
    echo -e "${RED}failed to generate results file${NC}"
fi

echo ""