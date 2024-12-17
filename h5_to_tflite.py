import tensorflow as tf
from keras.models import load_model

# 載入訓練好的 Keras 模型
model = load_model("fruit.keras")

# 創建一個轉換器
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 轉換模型為 TensorFlow Lite 格式
tflite_model = converter.convert()

# 將轉換後的模型保存到檔案
with open("fruit_lite_model.tflite", "wb") as f:
    f.write(tflite_model)