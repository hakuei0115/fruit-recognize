import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.applications import EfficientNetB0
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# 載入預處理後的數據
data = np.load("output_data.npz")
images = data['images']
labels = data['labels']

# 將標籤轉換為熱編碼（one-hot encoding）
num_classes = len(np.unique(labels))
labels = tf.keras.utils.to_categorical(labels, num_classes)

# 分割數據集為訓練集和驗證集
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# 創建EfficientNet-B0模型（不包含頂部的全連接層）
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 添加自定義頂部全連接層
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# 定義完整的模型
model = Model(inputs=base_model.input, outputs=predictions)

# 凍結EfficientNet-B0的層
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
print(f'損失率：{loss:.4f}')
print(f'準確度：{accuracy * 100:.2f}%')
