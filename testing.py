import pickle
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical

def test_model(model_path, data_path):
    # 載入預處理後的數據
    with open(data_path, "rb") as file:
        data = pickle.load(file)

    X_test = data['X_test']  # 使用測試集的圖片
    labels_test = data['y_test']  # 使用測試集的標籤

    # 載入模型
    loaded_model = load_model(model_path)

    # 創建標籤到整數的映射
    label_to_int = {label: i for i, label in enumerate(np.unique(labels_test))}

    # 使用映射將測試數據的標籤轉換為整數
    int_labels_test = np.array([label_to_int[label] for label in labels_test])

    # 將整數標籤進行獨熱編碼
    one_hot_labels_test = to_categorical(int_labels_test, num_classes=len(label_to_int))

    # 使用測試數據進行評估
    test_loss, test_accuracy = loaded_model.evaluate(X_test, one_hot_labels_test)
    print(f'測試損失率：{test_loss:.4f}')
    print(f'測試準確度：{test_accuracy * 100:.2f}%')

# 使用示例
test_model("fruit.h5", "preprocessed_data.pkl")