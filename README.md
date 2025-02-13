# Thai Text Emotion Classification

โปรแกรมวิเคราะห์อารมณ์จากข้อความภาษาไทย โดยใช้ Machine Learning (Multinomial Naive Bayes)

## คุณสมบัติ
- วิเคราะห์อารมณ์จากข้อความภาษาไทย
- รองรับการจำแนกอารมณ์หลายประเภท เช่น ความสุข, ความรัก, ความโกรธ, ความกลัว, ความประหลาดใจ และความเศร้า
- ใช้ไลบรารี PyThaiNLP สำหรับการประมวลผลภาษาไทย

## การติดตั้ง

1. ติดตั้ง dependencies ที่จำเป็น:

```
bash
pip install pythainlp pandas scikit-learn
```

## โครงสร้างโปรเจค

```
project/
│
├── main.py # โค้ดหลักของโปรแกรม
│
└── dataset/ # โฟลเดอร์สำหรับเก็บข้อมูล
└── csv/ # ไฟล์ CSV สำหรับเทรนโมเดล
├── training.csv
├── Emotion_classify_Data.csv
└── thai_emotion_dataset_large.csv
```

## วิธีการใช้งาน

1. เตรียมไฟล์ข้อมูลสำหรับเทรน (CSV) ในโฟลเดอร์ `dataset/csv/`
2. รันโปรแกรม:

```
bash
python main.py
```

3. ป้อนข้อความที่ต้องการวิเคราะห์เมื่อโปรแกรมถาม

## รูปแบบไฟล์ข้อมูล
ไฟล์ CSV ที่ใช้ในการเทรนต้องมีคอลัมน์ดังนี้:
- คอลัมน์สำหรับข้อความ (text)
- คอลัมน์สำหรับอารมณ์ (emotion/label)

## ตัวอย่างการใช้งาน

```
bash
Enter text: ฉันมีความสุขมากวันนี้
Predicted emotion :: happy
```

## เทคโนโลยีที่ใช้
- Python
- PyThaiNLP
- pandas
- scikit-learn