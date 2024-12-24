import numpy as np
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.applications import MobileNetV3Large
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import pickle
import matplotlib.pyplot as plt
from model.Adaptive_Particle_Grey_Wolf_Optimization import APGWO
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 載入數據並進行預處理
def load_and_preprocess_data(file_path, test_size=0.2, random_state=42):
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    images, labels = data['X_train'], data['y_train']
    label_to_int = {label: i for i, label in enumerate(np.unique(labels))}
    int_labels = np.array([label_to_int[label] for label in labels])
    one_hot_labels = to_categorical(int_labels, num_classes=len(label_to_int))

    X_train, X_val, y_train, y_val = train_test_split(images, one_hot_labels, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val, label_to_int

# 創建 MobileNetV2 模型
def create_model(learning_rate, dense_neurons, input_shape, num_classes):
    base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:-20]:
        layer.trainable = True
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_neurons, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# 目標函數
def objective_function(params, label_to_int, X_train, y_train, X_val, y_val):
    learning_rate, dense_neurons = params
    dense_neurons = int(dense_neurons)
    model = create_model(learning_rate, dense_neurons, (224, 224, 3), num_classes=len(label_to_int))
    val_loss = train_model(model, X_train, y_train, X_val, y_val, epochs=5)
    
    return val_loss

# 訓練模型並返回驗證損失
def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=5):
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(X_val, y_val)
    )
    
    return history.history['val_loss'][-1]

# 完整訓練模型並繪製曲線
def train_final_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=200):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    start_time = time.time()  # 記錄模型訓練開始時間
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )
    
    end_time = time.time()  # 記錄模型訓練結束時間
    training_time = end_time - start_time
    print(f"最終訓練時間: {training_time:.2f} 秒")

    # 計算 FPS
    total_frames = X_train.shape[0] * epochs
    fps = total_frames / training_time
    print(f"FPS (Frames Per Second): {fps:.2f}")
    
    # 繪製學習曲線
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.show()
    
    return model, training_time, fps

# 模型評估並輸出報告
def evaluate_model(model, X_val, y_val, label_to_int):
    # 取得預測結果
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)  # 預測的類別
    y_true_classes = np.argmax(y_val, axis=1)   # 真實的類別

    # 轉換類別映射為標籤
    int_to_label = {v: k for k, v in label_to_int.items()}

    # 輸出分類報告 (Precision, Recall, F1-Score)
    print("分類報告：")
    report = classification_report(y_true_classes, y_pred_classes, target_names=[int_to_label[i] for i in range(len(int_to_label))])
    print(report)

    # 計算並繪製混淆矩陣
    print("混淆矩陣：")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=int_to_label.values(), yticklabels=int_to_label.values())
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title("Confusion Matrix")
    plt.show()

    return report, cm

# 主程式
if __name__ == "__main__":
    # 固定隨機種子
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 記錄總時間
    total_start_time = time.time()
    
    X_train, X_val, y_train, y_val, label_to_int = load_and_preprocess_data("preprocessed_data.pkl")
    
    bounds = np.array([
        [0.0001, 1e-5],
        [256, 1024]
    ])
    
    apgwo_start_time = time.time()
    optimizer = APGWO(
        obj_func=lambda params: objective_function(params, label_to_int, X_train, y_train, X_val, y_val),
        dim=2,
        pop_size=5,
        max_iter=10,
        bounds=bounds
    )
    
    best_params, best_score = optimizer.optimize()
    apgwo_end_time = time.time()
    
    print(f"最佳參數：學習率={best_params[0]}, Dense 神經元數={int(best_params[1])}")
    print(f"APGWO 耗時: {apgwo_end_time - apgwo_start_time:.2f} 秒")
    
    # 使用最佳參數進行完整訓練
    learning_rate, dense_neurons = best_params[0], int(best_params[1])

    # 最終訓練
    model = create_model(learning_rate, dense_neurons, (224, 224, 3), num_classes=len(label_to_int))
    
    model, training_time, fps = train_final_model(model, X_train, y_train, X_val, y_val)
    
    # 評估最終模型
    loss, accuracy = model.evaluate(X_val, y_val)
    model.save('fruit.keras')
    print(f'最終驗證損失：{loss:.4f}')
    print(f'最終驗證準確率：{accuracy * 100:.2f}%')

    # 模型評估指標
    evaluate_model(model, X_val, y_val, label_to_int)

    # 記錄總時間
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"總訓練時間 (包含 APGWO): {total_time:.2f} 秒")
    print(f"FPS (Frames Per Second): {fps:.2f}")