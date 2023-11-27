import cv2
import numpy as np
import tensorflow as tf

calories_table = {
    "apple": 46,
    "banana": 91,
    "guava": 37.3,
    "bell-fruit": 35.6, #蓮霧
    "grape": 00,
    "orange": 00,
}

# 計算每克水果的卡路里函式
def calculate_calories(fruit, weight):
    if fruit in calories_table:
        calories_per_gram = calories_table[fruit] / 100
        total_calories = calories_per_gram * weight
        return total_calories
    else:
        return "Can't find calories for this fruit.";

def Recognize_images():
    # 載入 TensorFlow Lite 模型
    interpreter = tf.lite.Interpreter(model_path="fruit_lite_model.tflite")
    interpreter.allocate_tensors()

    # 獲取輸入和輸出的張量
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 加載測試圖像
    image_path = "apple.jpg"  # 替換成您的測試圖像路徑
    image = cv2.imread(image_path)  # 使用圖像處理庫讀取圖片

    # 對圖片進行像素值的歸一化處理，使其在0到1之間
    image = image.astype("float32") / 255.0

    # 在進行預測之前，您可能還需要調整圖片的大小和通道數，以使其與模型的輸入相容
    # 例如，如果模型的輸入大小為 (224, 224, 3)，您可以使用以下方式調整圖片：
    image = cv2.resize(image, (224, 224))  # 調整圖片大小

    # 創建標籤到整數的映射（與訓練時相同）
    label_to_int = {label: i for i, label in enumerate(["apple", "banana", "guava", "grape", "bell_fruit", "orange"])}  # 請替換成您的類別標籤

    # 創建反向映射，將整數類別映射回原始標籤
    int_to_label = {i: label for label, i in label_to_int.items()}

    # 將圖片轉換為輸入張量的格式
    input_data = np.expand_dims(image, axis=0).astype(input_details[0]['dtype'])

    # 將輸入張量設定給 TensorFlow Lite Interpreter
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 執行推理
    interpreter.invoke()

    # 獲取輸出張量的結果
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 在預測結果中找到整數類別
    predicted_class = np.argmax(output_data, axis=1)

    # 使用反向映射找到原始標籤
    predicted_label = int_to_label[predicted_class[0]]

    return predicted_label

fake_weight = 150;

fruit_result = Recognize_images()

calories = calculate_calories(fruit_result, fake_weight)

print(f"The calories for {fruit_result} weighing {fake_weight} grams are: {calories} kcal")