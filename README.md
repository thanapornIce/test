โมเดลตรวจจับวัตถุอันตราย เช่น ปืน มีด และระเบิด โดยใช้ **YOLOv11** ของ Ultralytics และเทรนจากภาพที่ได้จาก Roboflow พร้อม GUI ที่สามารถโหลดภาพจากเครื่องและแสดงผลได้ทันที

---

## 📦 Dataset

- ใช้ชุดข้อมูลที่เทรนไว้จาก Roboflow 

## 🔧 การติดตั้งและติดตั้ง yolov11

1. สร้าง Virtual Environment (แนะนำ)
   ```bash
   python -m venv venv
   source venv/bin/activate  # หรือ venv\Scripts\activate บน Windows
````

ค้นว่า yolov11 ที่ github และทำตามขั้นตอนติดตั้งแพ็กเกจ ultralytics (ซึ่งรวมถึง YOLOv8 และเครื่องมืออื่น ๆ) ผ่าน pip ในสภาพแวดล้อม Python ที่มีเวอร์ชัน ตั้งแต่ 3.8 ขึ้นไป และ ติดตั้ง PyTorch เวอร์ชัน 1.8 หรือใหม่กว่า แล้ว

 ```bash
  pip install ultralytics
````



2. ติดตั้งไลบรารีที่จำเป็น:

   ```bash
   pip install ultralytics opencv-python pillow tkinter
   ```

---


3. รันคำสั่งฝึกโมเดล:

   ```bash
   yolo task=detect mode=train model=yolov11n.pt data=data.yaml epochs=30 imgsz=640
   ```

---

## ✅ ทดสอบโมเดล

```bash
yolo task=detect mode=val model=best.pt data=data.yaml
```

หรือรันการทำนายบนรูปภาพ:

```bash
yolo task=detect mode=predict model=best.pt source="path/to/image.jpg"
```

---

## 🖼️ GUI โปรแกรมตรวจจับภาพ

รัน GUI ด้วย Python:

```bash
python detect.py
```

* เลือกภาพจากเครื่อง
* ระบบจะแสดงกรอบรอบวัตถุที่ตรวจพบ เช่น มีดหรือปืน

---

## 📂 โครงสร้างโปรเจกต์

```
├── dataset/              # ชุดข้อมูล (จาก Roboflow)
├── best.pt               # โมเดลที่ฝึกแล้ว
├── data.yaml             # ค่ากำหนดสำหรับ YOLO
├── train.py              # สคริปต์ฝึก (ถ้ามี)
└── README.md             # คู่มือการใช้งาน
```

