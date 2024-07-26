import json
import os
import re
import jieba
import random
from tqdm import tqdm

def clean_text(raw):
    raw = raw.strip()
    pattern = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fa5]+")
    return pattern.sub(' ', raw)

def load_data(jsonl_file, test_ratio=0.1):
    train_data, test_data = [], []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            data = json.loads(line)
            cleaned_text = clean_text(data['text'])
            processed_line = (data['label'], ' '.join(jieba.cut(cleaned_text)))
            if random.random() < test_ratio:
                test_data.append(processed_line)
            else:
                train_data.append(processed_line)
    return train_data, test_data

def save_data_to_fasttext_format(file_path, data_turple_list):
    with open(file_path, 'w', encoding='utf-8') as f:
        for label, text in data_turple_list:
            f.write("__label__"+str(label)+"\t"+text+"\n")

def load_labels_and_texts_from_txtfile(file_path):
    labels, texts = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                label, text = parts
                if label.startswith("__label__"):
                    labels.append(label.replace("__label__", ""))
                    texts.append(text)
    return labels, texts
