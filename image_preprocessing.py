import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

def augment_image(image):
    # 隨機水平翻轉
    if random.random() < 0.5:
        image = cv2.flip(image, 1)  # 水平翻轉

    # 隨機旋轉
    if random.random() < 0.5:
        angle = random.uniform(-30, 30)  # 隨機旋轉角度 (-30 到 30 度)
        rows, cols, _ = image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image = cv2.warpAffine(image, M, (cols, rows))

    # 隨機調整亮度
    if random.random() < 0.5:
        factor = random.uniform(0.5, 1.5)  # 隨機亮度調整因子 (0.5 到 1.5)
        image = cv2.convertScaleAbs(image, alpha=factor, beta=0)  # alpha: 對比度, beta: 亮度偏移

    return image


def preprocess_and_split_data(main_folder, target_size=(224, 224), test_size=0.2, validation_size=0.2, output_file="preprocessed_data.pkl"):
    labels = []  # 用於儲存標籤
    images = []  # 用於儲存預處理後的圖片

    # 遍歷主資料夾，顯示進度條
    for label in tqdm(os.listdir(main_folder), desc="處理標籤資料夾"):
        label_folder = os.path.join(main_folder, label)

        # 確保子資料夾是目錄
        if not os.path.isdir(label_folder):
            continue

        # 對每個標籤資料夾中的圖片進行預處理
        for image_filename in os.listdir(label_folder):
            # 檢查文件是否為jpg格式
            if image_filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(label_folder, image_filename)

                try:
                    # 讀取圖片
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"跳過無法讀取的圖片: {image_path}")
                        continue

                    # 調整大小
                    image = cv2.resize(image, target_size)

                    # 數據增強
                    image = augment_image(image)

                    # 標準化圖片（將像素值縮放到0到1的範圍）
                    image = image.astype("float32") / 255.0

                    # 儲存標籤和圖片
                    labels.append(label)
                    images.append(image)

                except Exception as e:
                    print(f"處理圖片時發生錯誤: {image_path}, 錯誤: {e}")
                    continue

    # 將數據分成訓練集、驗證集和測試集
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)

    # 將預處理後的數據儲存為檔案
    data = {
        "X_train": np.array(X_train), "y_train": np.array(y_train),
        "X_val": np.array(X_val), "y_val": np.array(y_val),
        "X_test": np.array(X_test), "y_test": np.array(y_test)
    }

    with open(output_file, "wb") as file:
        pickle.dump(data, file)

    # 輸出完成報告
    print(f"預處理完成: 總共處理了 {len(images)} 張圖片，分為:")
    print(f"訓練集: {len(X_train)}，驗證集: {len(X_val)}，測試集: {len(X_test)}")


# 指定主資料夾的路徑
main_folder = "training"

# 執行預處理並分割數據集
preprocess_and_split_data(main_folder, test_size=0.2, validation_size=0.2)
