# python 06-distilabel_medical_with_qa.py --url http://127.0.0.1:8000/v1 --dataset c00cjz00/Medical-R1-Distill-Data --datasetconfig default --datasetsplit train --questioncolumn question --answercolumn response --model c00cjz00/phi-4-14b-it-offon-R1-m22k --page 5 --pagesize 1024 --batchsize 64 --temperature 0.6 --maxnewtokens 4096 --template CUSTOM_TEMPLATE03
# python 06-distilabel_medical_with_qa.py --url http://127.0.0.1:8000/v1 --dataset c00cjz00/Medical-R1-Distill-Data --questioncolumn question --answercolumn response --model c00cjz00/phi-4-14b-it-offon-R1-m22k --page 5 --pagesize 1024 
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


#後設認知思考，簡單來說，就是思考你自己的思考過程。它是一種對自己的認知活動進行監控和調節的能力，包括理解自己的思考方式、評估自己的知識狀態，以及在必要時調整自己的思考策略
# 請手動修改{{ question }}{{ question }} 為你資料欄位名稱
CUSTOM_TEMPLATE01 = '''
You will be given a problem with a reference answer. Please analyze the problem step by step and provide your final answer from a "diagnostician's perspective", while following these guidelines: 
**(1) Response Quality**: Ensure **logical, well-structured, and comprehensive** responses, use **markdown formatting** for clarity, and acknowledge uncertainties when necessary. 
**(2) Ethical Operation**: **Refuse** illegal, violent, or explicit content, maintain **political neutrality**, and protect **user privacy** by avoiding data collection. 
**(3) Specialized Processing**: Before responding, perform internal reasoning within <think>...</think> tags, intermediate steps, and deductions are enclosed within these tags. Only provide the final response outside of '<think>...</think>'.  
**(4) Response Execution**: **Do not introduce yourself** or mention the response creator—simply **answer the question** following these rules."
**(5) Metacognitive Thinking**: Using Metacognitive Thinking to check your answer, based on the following steps: "Self-Awareness", "Reflection", "Self-Assessment", "Strategy Application"

### Medical Case:  
{{ question }}"

<think>
{{ response }}
</think>
'''.rstrip()


## 假設性思考 (Hypothetical thinking)：當對某種情況進行推理，例如「如果這位病人有糖尿病，那麼空腹八小時後的血糖值應該會超過 126 mg/dl，或者如果開立降血糖藥物，血糖值應該會改善」。這代表臨床思考中會建立假設並驗證。
# 請手動修改{{ question }}{{ question }} 為你資料欄位名稱
CUSTOM_TEMPLATE02 = '''
You will be given a problem with a reference answer. Please analyze the problem step by step and provide your final answer from a "diagnostician's perspective", while following these guidelines: 
**(1) Response Quality**: Ensure **logical, well-structured, and comprehensive** responses, use **markdown formatting** for clarity, and acknowledge uncertainties when necessary. 
**(2) Ethical Operation**: **Refuse** illegal, violent, or explicit content, maintain **political neutrality**, and protect **user privacy** by avoiding data collection. 
**(3) Specialized Processing**: Before responding, perform internal reasoning within <think>...</think> tags, intermediate steps, and deductions are enclosed within these tags. Only provide the final response outside of '<think>...</think>'.  
**(4) Response Execution**: **Do not introduce yourself** or mention the response creator—simply **answer the question** following these rules."
**(5) Hypothetical thinking**: Using hypothetical thinking to solve the problem, based on the following steps:"Generating Initial Hypotheses", "Testing Hypotheses and Gathering Further Information", "Exploring Alternative Diagnoses", "Evaluating Options"

### Medical Case:  
{{ question }}"

<think>
{{ response }}
</think>
'''.rstrip()



# 反事實思考 (Counterfactual reasoning)指的是思考如果情況與實際發生的不同時，可能會出現什麼結果。這是一種有意識且反思性的「系統二」思考
# 請手動修改{{ question }}{{ question }} 為你資料欄位名稱
CUSTOM_TEMPLATE03 = '''
You will be given a problem with a reference answer. Please analyze the problem step by step and provide your final answer from a "diagnostician's perspective", while following these guidelines: 
**(1) Response Quality**: Ensure **logical, well-structured, and comprehensive** responses, use **markdown formatting** for clarity, and acknowledge uncertainties when necessary. 
**(2) Ethical Operation**: **Refuse** illegal, violent, or explicit content, maintain **political neutrality**, and protect **user privacy** by avoiding data collection. 
**(3) Specialized Processing**: Before responding, perform internal reasoning within <think>...</think> tags, intermediate steps, and deductions are enclosed within these tags. Only provide the final response outside of '<think>...</think>'.  
**(4) Response Execution**: **Do not introduce yourself** or mention the response creator—simply **answer the question** following these rules."
**(5) Counterfactual reasoning**: Using counterfactual reasoning to solve the problem, based on the following steps:"Identify the Actual Clinical Event", "Select a Specific Element or Decision Point to Alter", "Construct the Counterfactual Scenario", "Trace the Potential Consequences of the Altered Scenario", "Compare the Hypothetical answer with the Actual answer"

### Medical Case:  
{{ question }}"

<think>
{{ response }}
</think>
'''.rstrip()


# Dual-Process Theories(雙重歷程理論) 是一種認知模型，它提出人類的思考和決策是透過兩種不同的系統或歷程運作的。這兩種系統通常被稱為系統一 (System 1) 和系統二 (System 2)
# 請手動修改{{ question }}{{ question }} 為你資料欄位名稱
CUSTOM_TEMPLATE04 = '''
You will be given a problem with a reference answer. Please analyze the problem step by step and provide your final answer from a "diagnostician's perspective", while following these guidelines: 
**(1) Response Quality**: Ensure **logical, well-structured, and comprehensive** responses, use **markdown formatting** for clarity, and acknowledge uncertainties when necessary. 
**(2) Ethical Operation**: **Refuse** illegal, violent, or explicit content, maintain **political neutrality**, and protect **user privacy** by avoiding data collection. 
**(3) Specialized Processing**: Before responding, perform internal reasoning within <think>...</think> tags, intermediate steps, and deductions are enclosed within these tags. Only provide the final response outside of '<think>...</think>'.  
**(4) Response Execution**: **Do not introduce yourself** or mention the response creator—simply **answer the question** following these rules."
**(5) Dual-Process Theories**: Using dual-Process Theories to solve the problem, based on the following steps: "System 1 Processing - Recognition or Non-Recognition", "further information gathering", "developing a differential diagnosis", "deductive reasoning - testing hypotheses", "inductive reasoning - assessing probabilities", "considering alternatives"

### Medical Case:  
{{ question }}"

<think>
{{ response }}
</think>
'''.rstrip()

#Naturalistic Decision Making (自然情境決策, NDM) 強調在真實世界、複雜、時間壓力大且充滿不確定性的環境中，人們如何做出決策，尤其是在像急診室這樣的臨床情境中。與傳統的分析性決策模型不同，NDM 更側重於經驗、直覺和情境感知
# 請手動修改{{ question }}{{ question }} 為你資料欄位名稱
CUSTOM_TEMPLATE05 = '''
You will be given a problem with a reference answer. Please analyze the problem step by step and provide your final answer from a "emergency physician's perspective", while following these guidelines: 
**(1) Response Quality**: Ensure **logical, well-structured, and comprehensive** responses, use **markdown formatting** for clarity, and acknowledge uncertainties when necessary. 
**(2) Ethical Operation**: **Refuse** illegal, violent, or explicit content, maintain **political neutrality**, and protect **user privacy** by avoiding data collection. 
**(3) Specialized Processing**: Before responding, perform internal reasoning within <think>...</think> tags, intermediate steps, and deductions are enclosed within these tags. Only provide the final response outside of '<think>...</think>'.  
**(4) Response Execution**: **Do not introduce yourself** or mention the response creator—simply **answer the question** following these rules."
**(5) Naturalistic Decision Making**: Using naturalistic decision making to solve the problem, based on the following steps: "Situation Assessment and Pattern Recognition", "Workable Solution Generation and Evaluation", "Acting, Interpreting, and Cultivating", "Sensemaking(using mindmap)"

### Medical Case:  
{{ question }}"

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
