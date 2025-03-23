import pandas as pd
import json

# 假設你的 Parquet 檔案是 messages_data.parquet
df = pd.read_parquet('train-00000-of-00001.parquet')

# 定義一個函數來處理每筆訊息
def process_messages(messages):
    questions = []
    answers = []
    
    # 遍歷每一輪對話
    for message in messages:
        content = message.get('content', '')
        role = message.get('role', '')
        
        if role == 'user':  # 假設使用者的訊息是問題
            questions.append(content)
        elif role == 'assistant':  # 假設助手的訊息是回答
            answers.append(content)
    
    if len(questions) == 1:  # 單輪對話
        question = questions[0]
        answer = answers[0] if answers else None
    else:  # 多輪對話
        # 問題+回答，將所有問題和回答組合起來，但不包括最後一輪的回答
        question = "\n".join([q + "\n" + a for q, a in zip(questions[:-1], answers[:-1])]) if questions[:-1] else None
        
        # 最後一輪問題保留並加上 'Problem:'
        if questions:
            question += f"\n\nQuestion: {questions[-1]}"  # 加上 'Problem: ' 前綴
        
        answer = answers[-1] if answers else None  # 最後一輪的回答

    return question, answer

# 使用 apply() 方法處理每一行的 'messages' 欄位
df[['input', 'output']] = df['messages'].apply(lambda messages: pd.Series(process_messages(messages)))

# 將處理後的資料保存為新的 Parquet 檔案
df.to_parquet('processed_messages_data.parquet', index=False)

# 顯示成功訊息
print("處理後的資料已儲存為 'processed_messages_data.parquet'")
