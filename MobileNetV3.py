import numpy as np
import tensorflow as tf
import pickle
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.applications import MobileNetV3Large
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# 載入預處理後的數據
with open("preprocessed_data.pkl", "rb") as file:
    data = pickle.load(file)

images = data['X_train']  # 使用訓練集的圖片
labels = data['y_train']  # 使用訓練集的標籤

# 創建標籤到整數的映射
label_to_int = {label: i for i, label in enumerate(np.unique(labels))}

# 使用映射將標籤轉換為整數
int_labels = np.array([label_to_int[label] for label in labels])

# 將整數標籤進行獨熱編碼
one_hot_labels = to_categorical(int_labels, num_classes=len(label_to_int))

# 分割數據集為訓練集和驗證集
X_train, X_val, y_train, y_val = train_test_split(images, one_hot_labels, test_size=0.2, random_state=42)

# 創建MobileNetV3模型（不包含頂部的全連接層）
base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 添加自定義頂部全連接層
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # 隱藏層
predictions = Dense(len(label_to_int), activation='softmax')(x)  # 輸出層

# 定義完整的模型
model = Model(inputs=base_model.input, outputs=predictions)

# 凍結MobileNetV3的層
for layer in base_model.layers:
    layer.trainable = False

# 編譯模型
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
batch_size = 32
epochs = 10

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

# 顯示損失率和準確度
loss, accuracy = model.evaluate(X_val, y_val)
model.save("fruit_mobilenetv3_large.h5")
print(f'損失率：{loss:.4f}')
print(f'準確度：{accuracy * 100:.2f}%')