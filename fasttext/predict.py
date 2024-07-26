import fasttext
import os
import json
import jieba
from utils.config import Config
from utils.preprocess import clean_text
from tqdm import tqdm
from collections import defaultdict

def load_model(config):
    model_path = config.get('model_path')
    return fasttext.load_model(model_path)

def predict_text(model, text):
    text_cut = ' '.join(jieba.cut(clean_text(text)))
    label = model.predict(text_cut)[0][0].replace('__label__', '')
    return label

def process_data(item, model, label_count):
    text = item.get('text', item.get('content', ''))
    label = predict_text(model, text)
    
    # 更新统计信息
    label_count[label] += 1
    return json.dumps({"label": label, "text": text}, ensure_ascii=False)

def predict_f2f(config, file_path):
    model = load_model(config)
    run_name = config.get('run_name')

    folder_path = os.path.dirname(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # 初始化统计信息
    label_count = defaultdict(int)
    total_count = 0

    with open(file_path, 'r', encoding='utf-8') as input_file:
        if file_path.endswith(".jsonl"):
            data_iter = (json.loads(line) for line in input_file)
        elif file_path.endswith(".json"):
            data_iter = json.load(input_file)
        else:
            raise ValueError("文件格式不支持，仅支持.json和.jsonl文件。")

        with open(os.path.join(folder_path, f'{file_name}_output.jsonl'), 'w', encoding='utf-8') as output_file:
            for item in tqdm(data_iter):
                output = process_data(item, model, label_count)
                total_count += 1
                output_file.write(output + '\n')

    # 写入日志文件
    with open(os.path.join(folder_path, f'{file_name}_{run_name}_log.txt'), 'w', encoding='utf-8') as log_file:
        log_file.write(f"总数据条数: {total_count}\n")
        for label, count in label_count.items():
            log_file.write(f"标签 {label}: {count} 条\n")

def predict(config, text):
    model = load_model(config)
    return predict_text(model, text)