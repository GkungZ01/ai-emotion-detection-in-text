# 🎭 Emotion Detection in Text

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

โปรแกรมวิเคราะห์อารมณ์จากข้อความภาษาไทย โดยใช้ Machine Learning (Multinomial Naive Bayes)

</div>

## ✨ คุณสมบัติ

- 🔍 วิเคราะห์อารมณ์จากข้อความภาษาไทย
- 🎯 รองรับการจำแนกอารมณ์ 6 ประเภท
- 🚀 ใช้ PyThaiNLP สำหรับการประมวลผลภาษาไทย
- ⚡ API พร้อมใช้งานผ่าน FastAPI

## 🎯 อารมณ์ที่รองรับ

| อารมณ์ | คำอธิบาย |
|--------|----------|
| 😢 Sadness | ความเศร้า |
| 😊 Happy | ความสุข |
| ❤️ Love | ความรัก |
| 😠 Anger | ความโกรธ |
| 😨 Fear | ความกลัว |
| 😲 Surprise | ความประหลาดใจ |

## 🚀 การติดตั้ง

1. Clone repository:

```bash
git clone https://github.com/yourusername/thai-emotion-detection.git
cd thai-emotion-detection
```

2. ติดตั้ง dependencies:

```bash
pip install -r requirements.txt
```

## 📁 โครงสร้างโปรเจค

```text
project/
├── 📜 main.py          # โค้ดหลักของโปรแกรม
├── 📜 requirements.txt  # รายการ dependencies
└── 📂 dataset/         # โฟลเดอร์สำหรับเก็บข้อมูล
    └── 📂 csv/         # ไฟล์ CSV สำหรับเทรนโมเดล
        ├── training.csv
        ├── Emotion_classify_Data.csv
        └── thai_emotion_dataset_large.csv
```

## 💻 วิธีการใช้งาน

### การใช้งานผ่าน API

3. รัน server:

```bash
python main.py
```

4. เรียกใช้ API:

```bash
curl -X GET "http://localhost:9959/?text=base64encodedtext"
```

### รูปแบบข้อมูลสำหรับเทรนโมเดล

ไฟล์ CSV ต้องประกอบด้วยคอลัมน์:

- `text`: ข้อความภาษาไทย
- `emotion`: ประเภทอารมณ์

## 🎯 ตัวอย่างการใช้งาน

### Python

```python
import requests
import base64

text = "ฉันมีความสุขมากวันนี้"
encoded_text = base64.b64encode(text.encode()).decode()
response = requests.get(f"http://localhost:9959/?text={encoded_text}")
print(f"Emotion: {response.json()['result']}")
# Output: Emotion: happy
```

### JavaScript

```javascript
const text = "ฉันมีความสุขมากวันนี้";
const encodedText = btoa(text);

fetch(`http://localhost:9959/?text=${encodedText}`)
  .then(response => response.json())
  .then(data => console.log(`Emotion: ${data.result}`))
  .catch(error => console.error('Error:', error));
// Output: Emotion: happy
```

## 🛠 เทคโนโลยีที่ใช้

- [Python](https://www.python.org/) - ภาษาหลักในการพัฒนา
- [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) - ไลบรารีประมวลผลภาษาไทย
- [scikit-learn](https://scikit-learn.org/) - ไลบรารี Machine Learning
- [FastAPI](https://fastapi.tiangolo.com/) - Web Framework

---
<div align="center">
Made with ❤️ in Thailand
</div>
