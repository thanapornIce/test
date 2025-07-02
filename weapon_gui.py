import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# โหลดโมเดล
model = load_model("detector_model.keras", compile=False)
labels = ['ปืน', 'มีด', 'ระเบิด']

def detect_weapon(img_path, model, window_size=64, step=32, threshold=0.9):
    img_color = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    for y in range(0, h - window_size + 1, step):
        for x in range(0, w - window_size + 1, step):
            patch = img_gray[y:y+window_size, x:x+window_size]
            resized = cv2.resize(patch, (64, 64)).reshape(1, 64, 64, 1) / 255.0
            pred = model.predict(resized, verbose=0)
            confidence = np.max(pred)
            if confidence > threshold:
                label = labels[np.argmax(pred)]
                cv2.rectangle(img_color, (x, y), (x+window_size, y+window_size), (0, 0, 255), 2)
                cv2.putText(img_color, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return img_color

# ฟังก์ชันสำหรับ GUI
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("โปรแกรมตรวจจับปืนและมีด")
        self.root.geometry("800x600")

        self.img_path = None

        # ปุ่ม Input
        self.btn_input = tk.Button(root, text="Input", bg="gray", fg="yellow", width=10, command=self.load_image)
        self.btn_input.place(x=50, y=100)

        # ปุ่ม Detect
        self.btn_detect = tk.Button(root, text="Detect", bg="gray", fg="yellow", width=10, command=self.detect_image)
        self.btn_detect.place(x=50, y=150)

        # ช่องแสดง Path
        self.entry_path = tk.Entry(root, width=50)
        self.entry_path.place(x=150, y=100)

        # พื้นที่แสดงภาพ
        self.panel = tk.Label(root)
        self.panel.place(x=150, y=150)

    def load_image(self):
        self.img_path = filedialog.askopenfilename()
        self.entry_path.delete(0, tk.END)
        self.entry_path.insert(0, self.img_path)
        self.display_image(self.img_path)

    def detect_image(self):
        if self.img_path:
            result_img = detect_weapon(self.img_path, model)
            cv2.imwrite("detected.jpg", result_img)
            self.display_image("detected.jpg")

    def display_image(self, path):
        img = Image.open(path)
        img = img.resize((500, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.panel.configure(image=img_tk)
        self.panel.image = img_tk

# เรียกใช้งาน
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
