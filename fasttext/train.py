import fasttext
import os
from utils.config import Config
from utils.preprocess import save_data_to_fasttext_format, load_data

def train(config):
    train_data, test_data = load_data(config.get('dataset_path'))
    train_file = f'fasttext_dataset/{config.get("run_name")}_train.txt'
    test_file = f'fasttext_dataset/{config.get("run_name")}_test.txt'
    save_data_to_fasttext_format(train_file, train_data)
    save_data_to_fasttext_format(test_file, test_data)


    model = fasttext.train_supervised(train_file, epoch=config.get('num_epochs'), lr=config.get('learning_rate'), 
        wordNgrams=config.get('wordNgrams'), verbose=config.get('verbose'), minCount=config.get('minCount'), loss='softmax')

    model.save_model(config.get('model_path'))