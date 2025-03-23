# python 05-distilabel_with_qa.py --url http://127.0.0.1:8000/v1 --dataset c00cjz00/Medical-R1-Distill-Data --datasetconfig default --datasetsplit train --questioncolumn question --answercolumn response --model Qwen/QwQ-32B --page 5 --pagesize 64 --batchsize 64 --temperature 0.6 --maxnewtokens 4096 --template CUSTOM_TEMPLATE01
# python 05-distilabel_with_qa.py --url http://127.0.0.1:8000/v1 --dataset c00cjz00/Medical-R1-Distill-Data --questioncolumn question --answercolumn response --model Qwen/QwQ-32B --page 5 --pagesize 1024 
# cat cmd.sh | /home/c00cjz00/.local/bin/parallel -j 4

import argparse
import os
from dotenv import load_dotenv
from datasets import load_dataset
from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration
from distilabel.steps import StepResources

# 匯入 API 金鑰
load_dotenv()  # 載入環境變數
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 請依照您輸入的 questioncolumn, answercolumn 手動修改欄位名稱{{ question }}{{ response }}
CUSTOM_TEMPLATE01 = '''
You will be given a problem with a reference answer. Please analyze the problem step by step and provide your final answer in **Traditional Chinese (zh-TW) from a Taiwanese perspective** while following these guidelines: 
**(1) Identity & Compliance**: State that you are an **AI assistant** in your initial response and comply with the **Republic of China (ROC) laws and regulations**, including its data privacy requirements. 
**(2) Capability Scope**: Support both **Chinese and English** queries, acknowledge **real-time information limitations**, and provide **technical explanations** for AI-related questions when necessary. 
**(3) Response Quality**: Ensure **logical, well-structured, and comprehensive** responses, use **markdown formatting** for clarity, and acknowledge uncertainties when necessary. 
**(4) Ethical Operation**: **Refuse** illegal, violent, or explicit content, maintain **political neutrality**, and protect **user privacy** by avoiding data collection. 
**(5) Specialized Processing**: Before responding, perform internal reasoning within <think>...</think> tags. Ensure that all thought processes, intermediate steps, and deductions are enclosed within these tags. Only provide the final response outside of '<think>...</think>'.  
**(6) Response Execution**: **Do not introduce yourself** or mention the response creator—simply **answer the question** following these rules.
### Problem:  
{{ question }}

<think>
{{ response }}
</think>
'''.rstrip()


def run_pipeline(url, dataset, datasetconfig, datasetsplit, questioncolumn, answercolumn, model, page, pagesize, batchsize, temperature, maxnewtokens, templatename):
    # 根據傳入的模板名稱獲取具體模板內容
    template = globals().get(templatename, None)  # 這裡使用 globals() 查找對應的模板
    if template is None:
        raise ValueError(f"Template {templatename} not found.")
        
    pipeline_id = "page_"+  str(page)
    with Pipeline(name=pipeline_id) as pipeline:
        TextGeneration(
            llm=OpenAILLM(
                api_key=OPENAI_API_KEY,  # 使用環境變數讀取 API 金鑰
                model=model,
                base_url=url,
                generation_kwargs={"temperature": temperature, "max_new_tokens": maxnewtokens},  # 設定生成參數
                timeout=120,  # 請求逾時時間 (秒)
                max_retries=6,  # 最大重試次數
            ),
            #system_prompt="detailed thinking on",
            input_batch_size=batchsize,  # 每次處理的輸入批次大小
            template=template,  # 使用自訂的 Prompt 模板
            columns=[questioncolumn, answercolumn],
            num_generations=1,  # 每個輸入要產生的回應數量
            group_generations=False,  # 是否將多個生成結果分組 (預設為 False，每個生成的結果都是獨立的)
            resources=StepResources(replicas=1),  # 設定此步驟的副本數量 (提高並行處理能力)
        )

    # 載入測試資料集 FreedomIntelligence/medical-o1-reasoning-SFT
    #dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    dataset = load_dataset(dataset, datasetconfig, split=datasetsplit)
    page_size = pagesize
    start = (page - 1) * page_size
    end = min(start + page_size, len(dataset))
    dataset = dataset.select(range(start, end))

    # 執行處理流程並取得結果
    distiset = pipeline.run(dataset=dataset, dataset_batch_size=page_size, use_cache=False)

    # 將結果上傳至 Hugging Face Hub
    #distiset.push_to_hub(repo_id="c00cjz00/medical-o1-reasoning-SFT_no_answer4")


if __name__ == "__main__":
    # 設置命令列參數
    parser = argparse.ArgumentParser(description="Run the reasoning pipeline.")
    parser.add_argument(
        "--url", 
        type=str, 
        required=True, 
        default=r"http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        default=r"c00cjz00/Medical-R1-Distill-Data",
    )
    parser.add_argument(
        "--datasetconfig", 
        type=str, 
        default="default",
    )    
    parser.add_argument(
        "--datasetsplit", 
        type=str, 
        default="train",
    )      
    parser.add_argument(
        "--questioncolumn", 
        type=str, 
        required=True, 
        default="question",
    )
    parser.add_argument(
        "--answercolumn", 
        type=str, 
        required=True, 
        default="response",
    )    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        default=r"c00cjz00/phi-4-14b-it-offon-R1-m22k",
    )
    parser.add_argument(
        "--page", 
        type=int, 
        required=True, 
        default=1,
    )
    parser.add_argument(
        "--pagesize", 
        type=int, 
        required=True, 
        default=64,
    )
    parser.add_argument(
        "--batchsize", 
        type=int, 
        default=64,
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.6,
    )   
    parser.add_argument(
        "--maxnewtokens", 
        type=int, 
        default=4096,
    )    
    parser.add_argument(
        "--template",
        type=str,
        default="CUSTOM_TEMPLATE01",
    )  
    
    
    args = parser.parse_args()
    # 處理頁面範圍
    url = args.url
    dataset = args.dataset
    datasetconfig = args.datasetconfig
    datasetsplit = args.datasetsplit
    questioncolumn = args.questioncolumn
    answercolumn = args.answercolumn
    model = args.model    
    page = args.page
    pagesize = args.pagesize
    batchsize = args.batchsize
    temperature = args.temperature
    maxnewtokens = args.maxnewtokens
    templatename = args.template  # 將模板名稱傳遞給 run_pipeline
    
    # 執行 pipeline
    print(f"Running with batch size: {batchsize}, temperature: {temperature}, max_new_tokens: {maxnewtokens}")
    run_pipeline(url, dataset, datasetconfig, datasetsplit, questioncolumn, answercolumn, model, page, pagesize, batchsize, temperature, maxnewtokens, templatename)
