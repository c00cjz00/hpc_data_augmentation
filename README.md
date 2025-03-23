# 知識蒸餾 (Think) 生成   

## 安裝套件 (I)
```bash=
mkdir -p $HOME/uv
cd $HOME/uv
export PATH=$PATH:$HOME/.local/bin
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv data_augmentation --python 3.11 && source $HOME/uv/data_augmentation/bin/activate && uv pip install --upgrade pip
uv pip install "distilabel[hf-inference-endpoints]"
uv pip install python-dotenv openai opencc beautifulsoup4 Pillow huggingface-hub

```
## 安裝套件 (II)
```bash=
mkdir -p /work/$(whoami)/uv
cd /work/$(whoami)/uv
export PATH=$PATH:$HOME/.local/bin
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv data_augmentation --python 3.11 && source /work/$(whoami)/uv/data_augmentation/bin/activate && uv pip install --upgrade pip
uv pip install "distilabel[hf-inference-endpoints]"
uv pip install python-dotenv openai opencc beautifulsoup4 Pillow huggingface-hub
```
## 下載套件
```bash=
#mkdir -p $HOME/github/ 
#cd $HOME/github/ 

mkdir -p /work/$(whoami)/github/
cd /work/$(whoami)/github/ 

git clone https://github.com/c00cjz00/hpc_data_augmentation.git
```

## **編輯 .env 並登錄API KEY**
- https://build.nvidia.com/deepseek-ai/deepseek-r1 取得 nvidia-key
```bash
cd $HOME/github/open-r1-dataset
echo "OPENAI_API_KEY=sk-xxxx" >.env
```

## **編輯 登錄HF KEY**

```bash
source $HOME/uv/data_augmentation/bin/activate
huggingface-cli login
```


## **指令: vllm_server.sh 指令說明**

1. 啟動vllm server
```bash
./vllm_server.sh
```

2. 確認模型是否運轉
```
curl -X 'GET' "http://127.0.0.1:8000/v1/models" \
-H 'accept: application/json' -H "Authorization: Bearer sk1234" |jq


curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
-H "Authorization: Bearer sk1234" \
-H "Content-Type: application/json" \
-d '{ "model": "c00cjz00/gemma-3-27b-it-offon-R1-m22k", "messages": [{"role": "user", "content": "123"}], "temperature": 0.6 }'

```

## **指令一: 04-distilabel_with_Q.py 指令說明**

```bash
python 04-distilabel_with_Q.py \
	--url http://127.0.0.1:8000/v1 \
	--dataset c00cjz00/Medical-R1-Distill-Data \
	--datasetconfig default \
	--datasetsplit train \
	--questioncolumn question \
	--model Qwen/QwQ-32B \
	--page 5 \
	--pagesize 64 \
	--batchsize 64 \
	--temperature 0.6 \
	--maxnewtokens 4096 \
	--template CUSTOM_TEMPLATE01
```

## **指令二: 05-distilabel_with_qa.py  指令說明**

```bash
python 05-distilabel_with_qa.py \
	--url http://127.0.0.1:8000/v1 \
	--dataset c00cjz00/Medical-R1-Distill-Data \
	--datasetconfig default \
	--datasetsplit train \
	--questioncolumn question \
	--answercolumn response \
	--model Qwen/QwQ-32B \
	--page 5 \
	--pagesize 64 \
	--batchsize 64 \
	--temperature 0.6 \
	--maxnewtokens 4096 \
	--template CUSTOM_TEMPLATE01
```

## **指令三: 06-distilabel_medical_with_qa.py 指令說明**

```bash
python 06-distilabel_medical_with_qa.py \
	--url http://127.0.0.1:8000/v1 \
	--dataset c00cjz00/Medical-R1-Distill-Data \
	--datasetconfig default --datasetsplit train \
	--questioncolumn question \
	--answercolumn response \
	--model c00cjz00/phi-4-14b-it-offon-R1-m22k \
	--page 5 \
	--pagesize 1024 \
	--batchsize 64 \
	--temperature 0.6 \
	--maxnewtokens 4096 \
	--template CUSTOM_TEMPLATE03
```

## **指令四: 07-parallel_cmd.sh 指令說明**

```bash
bash 07-parallel_cmd.sh > parallel.sh
```



##  **指令五: 製作COT資料表**

```bash
python 01-processed_cot_data.py
```

##  **指令六: 製作shareGPT**

```bash
python 02-shareGPT.py
```

##  **指令七: 錯誤製作流程檔案移除**
```bash
bash 03-remove.sh
```