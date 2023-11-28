import cv2
import numpy as np
import tensorflow as tf

user_age = input("Your age: ")
user_gender = input("Your gender: ")
user_height = input("Your height in cm: ")
user_weight = input("Your weight in kg: ")
user_activity_level = input("Your activity level: ")

calories_table = {
    "apple": 46,
    "banana": 91,
    "guava": 37.3,
    "bell-fruit": 35.6, #蓮霧
    "grape": 62.7,
    "orange": 41.8,
    "pineapple": 55.8,
}

def calculate_BMR(age, gender, height, weight):
    if gender == "male":
        return 66 + (13.7 * float(weight)) + (5 * float(height)) - (6.8 * float(age))
    else:
        return 655 + (9.6 * float(weight)) + (1.8 * float(height)) - (4.7 * float(age))
    
def calculate_TDEE(BMR, activity_level):
    if activity_level == "sedentary":
        return round(BMR * 1.2)
    elif activity_level == "lightly_active":
        return round(BMR * 1.375)
    elif activity_level == "moderately_active":
        return round(BMR * 1.55)
    elif activity_level == "very_active":
        return round(BMR * 1.725)
    elif activity_level == "extremely_active":
        return round(BMR * 1.9)

# 計算每克水果的卡路里函式
def calculate_calories(fruit, weight):
    if fruit in calories_table:
        calories_per_gram = calories_table[fruit] / 100
        total_calories = calories_per_gram * weight
        return total_calories
    else:
        return "Can't find calories for this fruit.";

def Recognize_images():
    interpreter = tf.lite.Interpreter(model_path="fruit_lite_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_path = "apple.jpg" #這裡要替換成拍照跟讀取
    image = cv2.imread(image_path)
    image = image.astype("float32") / 255.0
    image = cv2.resize(image, (224, 224))

    label_to_int = {label: i for i, label in enumerate(["apple", "banana", "guava", "grape", "bell_fruit", "orange", "pineapple"])}
    int_to_label = {i: label for label, i in label_to_int.items()}

    input_data = np.expand_dims(image, axis=0).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=1)
    predicted_label = int_to_label[predicted_class[0]]

    return predicted_label

BMR = calculate_BMR(user_age, user_gender, user_height, user_weight)
totalDailyEnergyExpenditure = calculate_TDEE(BMR, user_activity_level)

fake_weight = 150; #這裡改成測重

fruit_result = Recognize_images()

calories = calculate_calories(fruit_result, fake_weight)

print(f"The recommended daily calorie intake is: {totalDailyEnergyExpenditure} kcal")
print(f"The calories for {fruit_result} weighing {fake_weight} grams are: {calories} kcal")
print(f"After eating this fruit, you need to take {totalDailyEnergyExpenditure - calories} kcal in daily calorie intake")