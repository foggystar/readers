import pandas as pd
import dill
def read_parquet():
    data = pd.read_parquet('raw/train-00000-of-00001.parquet')
    list = [data['problem'].values,
            ['aimo-validation-aime']*len(data['problem'].values)]
    return list

def read_jsonl():
    data = pd.read_json('raw/usaco_bronze.jsonl',lines=True)
    content = [msg[0]['content'] for msg in data['messages']]
    list = [content,
            ['usaco_bronze']*len(content)]
    return list

def read_json():
    data = pd.read_json('raw/hotpotqa_sentence_bert_filter.json')
    list= [data['question'].values,
            ['hotpotqa_sentence_bert_filter']*len(data['question'].values)]
    return list

def read_dill():
    with open('raw/data.dill', 'rb') as f:
        data = dill.load(f)
    
    # 转换为DataFrame并获取目标列
    content = data['content'].values  # 假设content是目标列名
    
    # 创建包含内容和标签的list
    list = [content,
            ['dill_source']*len(content)]
    return list

if __name__ == '__main__':
    res = list(zip(*read_jsonl())) \
        +list(zip(*read_parquet())) \
        +list(zip(*read_json()))
    print(len(res),len(res[0]))
    pd.DataFrame(res).to_csv('csv/problem.csv', index=False, header=['Problem','Source'])