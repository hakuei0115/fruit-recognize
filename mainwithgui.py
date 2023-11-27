import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk

calories_table = {
    "apple": 46,
    "banana": 91,
    "guava": 37.3,
    "bell-fruit": 35.6, #蓮霧
    "grape": 62.7,
    "orange": 41.8,
}

def calculate_BMR(age, gender, height, weight):
    if gender == "Male":
        return 66 + (13.7 * float(weight)) + (5 * float(height)) - (6.8 * float(age))
    else:
        return 655 + (9.6 * float(weight)) + (1.8 * float(height)) - (4.7 * float(age))
    pass

def calculate_TDEE(BMR, activity_level):
    if activity_level == "Sedentary":
        return round(BMR * 1.2)
    elif activity_level == "Lightly active":
        return round(BMR * 1.375)
    elif activity_level == "Moderately active":
        return round(BMR * 1.55)
    elif activity_level == "Very active":
        return round(BMR * 1.725)
    elif activity_level == "Extremely active":
        return round(BMR * 1.9)
    pass

# 計算每克水果的卡路里函式
def calculate_calories(fruit, weight):
    if fruit in calories_table:
        calories_per_gram = calories_table[fruit] / 100
        total_calories = calories_per_gram * weight
        return total_calories
    else:
        return "Can't find calories for this fruit.";
    pass

def Recognize_images():
    interpreter = tf.lite.Interpreter(model_path="fruit_lite_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_path = "apple.jpg" #這裡要替換成拍照跟讀取
    image = cv2.imread(image_path)
    image = image.astype("float32") / 255.0
    image = cv2.resize(image, (224, 224))

    label_to_int = {label: i for i, label in enumerate(["apple", "banana", "bell-fruit", "grape", "guava", "orange"])}
    int_to_label = {i: label for label, i in label_to_int.items()}

    input_data = np.expand_dims(image, axis=0).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=1)
    predicted_label = int_to_label[predicted_class[0]]

    return predicted_label
    pass

def submit_button_click():
    check = tk.Tk()
    check.title("Make sure fruit put on the scales")

    check_label = ttk.Label(check, text="Does the fruit on the scales?")
    check_label.grid(row=0, column=0) #TODO: 這裡要新增確認按鈕然後跳出結果
    pass

def on_calculate_button_click():
    user_age = entry_age.get()
    user_gender = gender_var.get()
    user_height = entry_height.get()
    user_weight = entry_weight.get()
    user_activity = activity_var.get()

    BMR = calculate_BMR(user_age, user_gender, user_height, user_weight)
    totalDailyEnergyExpenditure = calculate_TDEE(BMR, user_activity)

    fake_weight = 150; #這裡改成測重

    fruit_result = Recognize_images()

    calories = calculate_calories(fruit_result, fake_weight)

    root2 = tk.Tk()
    root2.title("Calorie Results")

    result_label = ttk.Label(root2, text="")
    result_label2 = ttk.Label(root2, text="")
    result_label3 = ttk.Label(root2, text="")

    result_label.config(text=f"The recommended daily calorie intake is: {totalDailyEnergyExpenditure} kcal")
    result_label2.config(text=f"The calories for {fruit_result} weighing {fake_weight} grams are: {calories} kcal")
    result_label3.config(text=f"After eating this fruit, you need to take {totalDailyEnergyExpenditure - calories} kcal in daily calorie intake")

    result_label.grid(row=6, column=0, columnspan=2, pady=5)
    result_label2.grid(row=7, column=0, columnspan=2, pady=5)
    result_label3.grid(row=8, column=0, columnspan=2, pady=5)
    pass

# 創建主視窗
root = tk.Tk()
root.title("Calorie Calculator")

# 創建 GUI 元件
label_age = ttk.Label(root, text="Your age:")
entry_age = ttk.Entry(root)

label_gender = ttk.Label(root, text="Your gender:")
gender_var = tk.StringVar()
male_radio = ttk.Radiobutton(root, text="Male", variable=gender_var, value="Male")
female_radio = ttk.Radiobutton(root, text="Female", variable=gender_var, value="Female")

label_height = ttk.Label(root, text="Your height in (cm):")
entry_height = ttk.Entry(root)

label_weight = ttk.Label(root, text="Your weight in (kg):")
entry_weight = ttk.Entry(root)

label_activity = ttk.Label(root, text="Your activity level:")
activity_var = tk.StringVar()
sedentary_radio = ttk.Radiobutton(root, text="幾乎不運動", variable=activity_var, value="Sedentary")
lightly_active_radio = ttk.Radiobutton(root, text="每週運動 1-3 天", variable=activity_var, value="Lightly active")
moderately_active_radio = ttk.Radiobutton(root, text="每週運動 3-5 天", variable=activity_var, value="Moderately active")
very_active_radio = ttk.Radiobutton(root, text="每週運動 6-7 天", variable=activity_var, value="Very active")
extremely_active_radio = ttk.Radiobutton(root, text="長時間運動或體力勞動工作", variable=activity_var, value="Extremely active")

# calculate_button = ttk.Button(root, text="Calculate TDEE And see fruit weight", command=on_calculate_button_click)
submit_button = ttk.Button(root, text="Calculate TDEE And put fruit on scales", command=submit_button_click)

# 放置 GUI 元件
label_age.grid(row=0, column=0, padx=5, pady=5)
entry_age.grid(row=0, column=1, padx=5, pady=5)

label_gender.grid(row=1, column=0, padx=5, pady=5)
male_radio.grid(row=1, column=1, padx=5, pady=5)
female_radio.grid(row=1, column=2, padx=5, pady=5)

label_height.grid(row=2, column=0, padx=5, pady=5)
entry_height.grid(row=2, column=1, padx=5, pady=5)

label_weight.grid(row=3, column=0, padx=5, pady=5)
entry_weight.grid(row=3, column=1, padx=5, pady=5)

label_activity.grid(row=4, column=0, padx=5, pady=5)
sedentary_radio.grid(row=4, column=1, padx=5, pady=5)
lightly_active_radio.grid(row=4, column=2, padx=5, pady=5)
moderately_active_radio.grid(row=4, column=3, padx=5, pady=5)
very_active_radio.grid(row=5, column=1, padx=5, pady=5)
extremely_active_radio.grid(row=5, column=2, padx=5, pady=5)

# calculate_button.grid(row=6, column=0, columnspan=5, pady=10)
submit_button.grid(row=6, column=0, columnspan=5, pady=10)

# 啟動主迴圈
root.mainloop()