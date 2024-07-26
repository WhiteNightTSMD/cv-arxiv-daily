from utils.config import Config
import os
import train
import evaluate
import predict

def main():
    config = Config.get_config()
    config.replace_run_name_placeholder()  # 替换配置文件中的占位符

    operation = config.get('operation')
    predict_text = config.get('predict_text')

    if operation == 'train':
        train.train(config)
        evaluate.evaluate_model(config)
        return None
        
    elif operation == 'predict':
        if predict_text and os.path.isfile(predict_text):
            return predict.predict_f2f(config, predict_text)
        elif predict_text:
            label = predict.predict(config, predict_text)
            print(label)
            return label
        else:
            print("无有效的预测输入。")
            return None
    else:
        print("未知操作。")
        return None

if __name__ == '__main__':
    main()