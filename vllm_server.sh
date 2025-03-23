#!/bin/bash

# 設定變數
#MODEL_NAME="c00cjz00/phi-4-14b-it-offon-R1-m22k"  # 你想要使用的模型名稱
#MODEL_NAME="c00cjz00/gemma-3-27b-it-offon-R1-m22k"
MODEL_NAME="Qwen/QwQ-32B"
PORT=8000             # 服務的端口
API_KEY="sk1234"          # 你的 API 金鑰

# 環境設定
#cd /work/$(whoami)/github/hpc_vllm
export PATH=$PATH:$HOME/.local/bin
mkdir -p ./home ./logs ./output

# 背景執行 Singularity 容器
ml singularity
singularity exec --nv --no-home \
    -B /work \
    -B ./home:$HOME \
    -B /work/c00cjz00/docker/vllm-openai_v0.8.1/opt:/opt \
    -B /work/c00cjz00/docker/vllm-openai_v0.8.1/root:/root \
    /work/c00cjz00/docker/vllm-openai_v0.8.1.sif \
    bash -c "vllm serve $MODEL_NAME \
        --trust-remote-code \
        --served-model-name $MODEL_NAME \
        --tensor-parallel-size 1 \
        --dtype auto \
        --port $PORT \
        --max-model-len 16384 \
        --max-seq-len-to-capture 16384 \
        --gpu-memory-utilization 0.97 \
        --swap-space 32 \
        --enforce-eager \
        --api-key $API_KEY \
    " > ./logs/vllm_output.log 2>&1 &

echo "vLLM 服務已在背景執行。PID: $!"
echo "輸出已重導向到 vllm_output.log"
