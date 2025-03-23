from datasets import load_dataset, DatasetDict
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# 1. 读取数据集
dataset_name = "FreedomIntelligence/Medical-R1-Distill-Data"
dataset = load_dataset(dataset_name)

def rename_columns(example):
    example["reasoning"] = example.pop("reasoning (reasoning_content)")
    example["response"] = example.pop("response (content)")
    return example



def convert_to_sharegpt_format(example):
    """转换为 ShareGPT 格式"""
    question = example["question"]
    reasoning = example["reasoning"]
    response = example["response"]
    
    conversations = [
        {"value": question, "from": "human"},
        {"value": f"<think>\n\n{reasoning}\n</think>\n\n{response}", "from": "gpt"}
    ]
    
    return {"conversations": conversations}

# 2. 转换数据
dataset = dataset.map(rename_columns)
dataset = dataset.map(convert_to_sharegpt_format)

#dataset = dataset.remove_columns(["question", "reasoning (reasoning_content)", "response (content)"])  # 移除原始列

dataset = DatasetDict({"train": dataset["train"]})  # 确保数据集格式

# 3. 直接保存为 Parquet 格式
data = [example for example in dataset["train"]]
df = pd.DataFrame(data)
table = pa.Table.from_pandas(df)
output_parquet = "medical_r1_sharegpt.parquet"
pq.write_table(table, output_parquet)

