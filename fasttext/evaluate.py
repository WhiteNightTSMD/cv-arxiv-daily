import fasttext
from sklearn.metrics import confusion_matrix, classification_report
from utils.config import Config
from utils.preprocess import load_labels_and_texts_from_txtfile

def evaluate_model(config):
    model_path = config.get('model_path')
    result_save_path = config.get('result_save_path')
    test_file = f'fasttext_dataset/{config.get("run_name")}_test.txt'

    model = fasttext.load_model(model_path)
    result = model.test(test_file)
    print("准确率:", result)

    labels, texts = load_labels_and_texts_from_txtfile(test_file)
    unique_labels = set(labels)
    print(f"unique labels: {unique_labels}")

    predicted_labels = [model.predict(text.replace('\n', ' '))[0][0].replace('__label__', '') for text in texts]

    conf_matrix = confusion_matrix(labels, predicted_labels)
    report = classification_report(labels, predicted_labels, output_dict=True)

    with open(result_save_path, 'w') as file:
        file.write("Classification Report:\n")
        file.write(str(report))
        file.write("\n\nConfusion Matrix:\n")
        file.write(str(conf_matrix))

    return report, conf_matrix