# Calorie Calculator

這是一個基於圖像識別的卡路里計算器，能夠識別水果並計算其卡路里含量。

## 功能

- 使用 HX711 進行重量檢測
- 使用 TensorFlow Lite 模型識別水果
- 計算水果的卡路里含量
- 計算基礎代謝率 (BMR) 和總每日能量消耗 (TDEE)

## 安裝指南

1. 直接 Download ZIP
2. 去到下載解壓縮，並找到使用VSCode打開該資料夾
3. 使用快捷鍵 Ctrl+shift+` 打開終端機
4. 於終端機輸入下列指令安裝虛擬環境
   1. pip install virtualenv
   2. virtualenv .venv <- 虛擬環境名稱 可隨意決定
   3. pip install -r requirements.txt <- 所有所需套件以及規定版本都放置在此文件檔中 使用此指令可一鍵安裝

## 目錄結構

```scss
    - fruit-recognize-master
    | - recognize_image 預測圖片放置區
    | - training 訓練圖片放置區
    | - .gitignore 上傳git時決定哪個不進行上傳的檔案
    | - h5_to_tflite.py 把訓練好的 Keras 模型轉成 TensorFlow Lite 格式
    | - image_preprocessing.py 預處理圖片的程式
    | - image_recognition.py 讀取模型並進行預測 顯示預測結果和機率
    | - main.py 主程式
    | - MobileNetV2.py 讀取圖片預處理後的檔案進行模型訓練 這裡採用MobileNet V2
    | - requirements.txt 管理套件版本的文件
    | - README.md 說明文件
```

## 使用說明

1. 首先確認 training 資料夾中有數據集，且不同的數據須放置在不同資料夾，資料夾名稱會作為標籤
2. 使用 `image_preprocessing.py` 對數據集進行預處理並把結果儲存成一個檔案
3. 使用 `MobileNetV2.py` 進行訓練並產生一個 Keras 模型
4. 使用 `h5_to_tflite.py` 把訓練好的 Keras 模型轉成 TensorFlow Lite 格式
5. 把要進行預測的圖片放到 recognize_image 預測圖片放置區
6. 使用 `image_recognition.py` 讀取模型並進行預測 顯示預測結果和機率 這裡可以用來判斷模型好壞
7. 使用 `main.py` 啟動 app 