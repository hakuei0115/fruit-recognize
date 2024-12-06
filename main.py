import cv2
import numpy as np
import tensorflow as tf
import ttkbootstrap as ttk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont

class CalorieCalculator:
    calories_table = {
        "apple": 46,
        "banana": 91,
        "guava": 37.3,
        "bell-fruit": 35.6, #蓮霧
        "grape": 62.7,
        "orange": 41.8,
        "pineapple": 55.8,
    }

    def calculate_BMR(self, age, gender, height, weight):
        if gender == "Male":
            return 66 + (13.7 * float(weight)) + (5 * float(height)) - (6.8 * float(age))
        else:
            return 655 + (9.6 * float(weight)) + (1.8 * float(height)) - (4.7 * float(age))
        pass

    def calculate_TDEE(self, BMR, activity_level):
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
    def calculate_calories(self, fruit, weight):
        if fruit in CalorieCalculator.calories_table:
            calories_per_gram = CalorieCalculator.calories_table[fruit] / 100
            total_calories = calories_per_gram * weight
            return round(total_calories)
        else:
            return "Can't find calories for this fruit."
        pass

def Recognize_images():
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # ret, frame = cap.read()

    # capture_path = "recognize_image/captured_image.jpg"
    # cv2.imwrite(capture_path, frame)

    # cap.release()

    interpreter = tf.lite.Interpreter(model_path="fruit_lite_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_path = "recognize_image/apple.jpg" #這裡要替換成讀取拍照結果
    image = cv2.imread(image_path)
    image = image.astype("float32") / 255.0
    image = cv2.resize(image, (224, 224))

    label_to_int = {label: i for i, label in enumerate(["apple", "banana", "bell-fruit", "grape", "guava", "orange", "pineapple"])}
    int_to_label = {i: label for label, i in label_to_int.items()}

    input_data = np.expand_dims(image, axis=0).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=1)

    predicted_probability = 0

    threshold = 0.85
    if output_data[0, predicted_class[0]] < threshold:
        predicted_label = "unknown"
    else:
        predicted_label = int_to_label[predicted_class[0]]
        predicted_probability = output_data[0, predicted_class[0]] * 100
    
    return predicted_label, predicted_probability

class CalorieCalculatorApp:
    def __init__(self, root, calculator):
        self.root = root
        self.calculator = calculator
        # 創建 GUI 元件
        label_age = ttk.Label(root, text="Your age:")
        self.entry_age = ttk.Entry(root, bootstyle="light")

        label_gender = ttk.Label(root, text="Your gender:")
        self.gender_var = ttk.StringVar()
        male_radio = ttk.Radiobutton(root, text="Male", variable=self.gender_var, value="Male")
        female_radio = ttk.Radiobutton(root, text="Female", variable=self.gender_var, value="Female")

        label_height = ttk.Label(root, text="Your height in (cm):")
        self.entry_height = ttk.Entry(root, bootstyle="light")

        label_weight = ttk.Label(root, text="Your weight in (kg):")
        self.entry_weight = ttk.Entry(root, bootstyle="light")

        label_activity = ttk.Label(root, text="Your activity level:")
        self.activity_var = ttk.StringVar()
        sedentary_radio = ttk.Radiobutton(root, text="幾乎不運動", variable=self.activity_var, value="Sedentary")
        lightly_active_radio = ttk.Radiobutton(root, text="每週運動 1-3 天", variable=self.activity_var, value="Lightly active")
        moderately_active_radio = ttk.Radiobutton(root, text="每週運動 3-5 天", variable=self.activity_var, value="Moderately active")
        very_active_radio = ttk.Radiobutton(root, text="每週運動 6-7 天", variable=self.activity_var, value="Very active")
        extremely_active_radio = ttk.Radiobutton(root, text="長時間運動或體力勞動工作", variable=self.activity_var, value="Extremely active")
        submit_button = ttk.Button(root, bootstyle=("primary", "outline-toolbutton"), text="Calculate TDEE And put fruit on scales", command=self.submit_button_click)

        # 設置列和行的權重
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=1)
        root.columnconfigure(3, weight=1)
        root.columnconfigure(4, weight=1)

        root.rowconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(2, weight=1)
        root.rowconfigure(3, weight=1)
        root.rowconfigure(4, weight=1)
        root.rowconfigure(5, weight=1)
        root.rowconfigure(6, weight=1)

        # 放置 GUI 元件
        label_age.grid(row=0, column=0, padx=5, pady=10, sticky="e")
        self.entry_age.grid(row=0, column=1, columnspan=4, padx=5, pady=10, sticky="we")

        label_gender.grid(row=1, column=0, padx=5, pady=10, sticky="e")
        male_radio.grid(row=1, column=1, padx=5, pady=10, sticky="w")
        female_radio.grid(row=1, column=2, padx=5, pady=10, sticky="w")

        label_height.grid(row=2, column=0, padx=5, pady=10, sticky="e")
        self.entry_height.grid(row=2, column=1, columnspan=4, padx=5, pady=10, sticky="we")

        label_weight.grid(row=3, column=0, padx=5, pady=10, sticky="e")
        self.entry_weight.grid(row=3, column=1, columnspan=4, padx=5, pady=10, sticky="we")

        label_activity.grid(row=4, column=0, padx=5, pady=10, sticky="e")
        sedentary_radio.grid(row=4, column=1, padx=5, pady=10, sticky="w")
        lightly_active_radio.grid(row=4, column=2, padx=5, pady=10, sticky="w")
        moderately_active_radio.grid(row=4, column=3, padx=5, pady=10, sticky="w")
        very_active_radio.grid(row=5, column=1, padx=5, pady=10, sticky="w")
        extremely_active_radio.grid(row=5, column=2, padx=5, pady=10, sticky="w")

        submit_button.grid(row=6, column=0, columnspan=5, pady=20, sticky="we")

    def submit_button_click(self):
        user_age = self.entry_age.get()
        user_gender = self.gender_var.get()
        user_height = self.entry_height.get()
        user_weight = self.entry_weight.get()
        user_activity = self.activity_var.get()

        if not user_age or not user_gender or not user_height or not user_weight or not user_activity:
            messagebox.showerror("Error", "Please fill in all required fields.")
            return False
        
        if not user_age.isdigit() or not user_height.isdigit() or not user_weight.isdigit():
            messagebox.showerror("Error", "Please enter valid numbers for age, height, and weight.")
            return False
        else:
            messagebox.showinfo("Put the fruit on the scales", "Make sure to put the fruit on the scales.")
            self.on_calculate_button_click()

    def on_calculate_button_click(self):
        user_age = self.entry_age.get()
        user_gender = self.gender_var.get()
        user_height = self.entry_height.get()
        user_weight = self.entry_weight.get()
        user_activity = self.activity_var.get()

        BMR = self.calculator.calculate_BMR(user_age, user_gender, user_height, user_weight)
        totalDailyEnergyExpenditure = self.calculator.calculate_TDEE(BMR, user_activity)

        fake_weight = 150; #這裡改成測重

        fruit_result, fruit_probability = Recognize_images()
        probability_text = f"{fruit_probability:.2f}%"

        root2 = ttk.Toplevel(self.root)
        root2.title("Calorie Results")
        root_x, root_y = self.root.winfo_x(), self.root.winfo_y()
        root2.geometry(f"+{root_x + 50}+{root_y}")

        result_label = ttk.Label(root2, text="")
        result_label2 = ttk.Label(root2, text="")
        result_label3 = ttk.Label(root2, text="")

        image_path = "recognize_image/apple.jpg" #修改成讀取拍照的路徑
        image = Image.open(image_path)
        image = image.resize((500, 500), Image.Resampling.LANCZOS)

        draw = ImageDraw.Draw(image)
        default_font = ImageFont.load_default()
        font_size = 36
        font = default_font.font_variant(size=font_size)
        if fruit_result == "unknown":
            draw.text((10, 10), fruit_result, fill="red", font=font)
        else:
            draw.text((10, 10), fruit_result, fill="red", font=font)
            draw.text((200, 10), probability_text, fill="red", font=font)
            calories = self.calculator.calculate_calories(fruit_result, fake_weight)

        photo = ImageTk.PhotoImage(image)
        image_label = ttk.Label(root2, image=photo)
        image_label.image = photo
        image_label.grid(row=0, column=0, columnspan=2, padx=5, pady=20, sticky="nsew")

        if fruit_result == "unknown":
            result_label.config(text=f"The recommended daily calorie intake is: {totalDailyEnergyExpenditure} kcal")
            result_label2.config(text=f"Can't recognize this fruit, but we can give you the weight is {fake_weight} grams")
            result_label3.config(text=f"And you can search '100克的水果等於多少卡路里' to calculate.")
        else:
            result_label.config(text=f"The recommended daily calorie intake is: {totalDailyEnergyExpenditure} kcal")
            result_label2.config(text=f"The calories for {fruit_result} weighing {fake_weight} grams are: {calories} kcal")
            result_label3.config(text=f"After eating this fruit, you need to take {totalDailyEnergyExpenditure - calories} kcal in daily calorie intake")

        # 設置列和行的權重
        root2.columnconfigure(0, weight=1)
        root2.columnconfigure(1, weight=1)
        root2.rowconfigure(0, weight=1)
        root2.rowconfigure(1, weight=1)
        root2.rowconfigure(2, weight=1)
        root2.rowconfigure(3, weight=1)
        
        result_label.grid(row=1, column=0, columnspan=2, padx=5, pady=20, sticky="nsew")
        result_label2.grid(row=2, column=0, columnspan=2, padx=5, pady=20, sticky="nsew")
        result_label3.grid(row=3, column=0, columnspan=2, padx=5, pady=20, sticky="nsew")

if __name__ == "__main__":
    calculator = CalorieCalculator()
    root = ttk.Window(
        title = "Calorie Calculator",
        themename = "superhero",
        position=(800, 400),
        resizable=None,
        alpha=1.0,
    )
    root.geometry("1080x800")
    app = CalorieCalculatorApp(root, calculator)
    root.mainloop()