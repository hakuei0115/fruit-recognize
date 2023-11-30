import cv2
import numpy as np
from keras.models import load_model

# 載入訓練好的模型
model = load_model("fruit.h5")

# 加載測試圖像
image_path = "recognize_image/captured_image.jpg"  # 替換成您的測試圖像路徑
image = cv2.imread(image_path)  # 使用圖像處理庫讀取圖片

# 對圖片進行像素值的歸一化處理，使其在0到1之間
image = image.astype("float32") / 255.0

# 在進行預測之前，您可能還需要調整圖片的大小和通道數，以使其與模型的輸入相容
# 例如，如果模型的輸入大小為(224, 224, 3)，您可以使用以下方式調整圖片：
image = cv2.resize(image, (224, 224))  # 調整圖片大小

# 進行模型預測
predictions = model.predict(np.expand_dims(image, axis=0))

# 創建標籤到整數的映射（與訓練時相同）
label_to_int = {label: i for i, label in enumerate(["apple", "banana", "bell-fruit", "grape", "guava", "orange", "pineapple"])}  # 請替換成您的類別標籤

# 創建反向映射，將整數類別映射回原始標籤
int_to_label = {i: label for label, i in label_to_int.items()}

# 在預測結果中找到整數類別（假設 predictions 是模型的預測結果）
predicted_class = np.argmax(predictions, axis=1)

# 使用反向映射找到原始標籤
predicted_label = int_to_label[predicted_class[0]]

# 顯示預測的類別和機率
print(f"Predicted class: {predicted_label}")
print("Prediction probabilities:")
for label, prob in zip(int_to_label.values(), predictions[0]):
    prob_percentage = prob * 100
    print(f"{label}: {prob_percentage:.2f}%")

threshold = 0.85  # 設置預測標籤的門檻值
if predictions[0, predicted_class[0]] < threshold:
    predicted_label = "Unknown"

# 在圖像上繪製預測結果
cv2.putText(image, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 顯示帶有預測標籤的圖像
cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
