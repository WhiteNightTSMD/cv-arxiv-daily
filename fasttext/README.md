# text_classification
中英文分类任务，添加各种模型支持ing

# fasttext
'''
pip install fasttext
'''
## 训练 
python main.py --operation train --run_name {注意修改config.json的路径}
训练集路径请自行更改或者加 --dataset_path 参数
## 预测

python main.py --operation predict --run_name *** --predict_text {json文件路径或者你要预测的文本} --config {config路径}

### 示例,预测单个文本
python main.py --operation predict --run_name quality_judge --predict_text 你叫什么名字？

### 示例,预测整个文件并输出

python main.py --operation predict --run_name quality_judge --predict_text /data/*/预测文件.json