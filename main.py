# นำเข้าไลบรารีที่จำเป็น
# pythainlp: สำหรับประมวลผลภาษาไทย
# pandas: สำหรับจัดการข้อมูลในรูปแบบตาราง
# base64: สำหรับเข้ารหัสและถอดรหัสข้อความ
# FastAPI: สำหรับสร้าง Web API
# CountVectorizer: สำหรับแปลงข้อความเป็นเวกเตอร์
# MultinomialNB: โมเดล Naive Bayes สำหรับการจำแนกข้อความ
import pythainlp
import pandas as pd
import base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import uvicorn

# สร้าง FastAPI application instance
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# สร้างลิสต์สำหรับเก็บข้อความและอารมณ์
texts = []
labels = []


def importDataSetCsv(filename: str, text_column: str, emotions_column: str, emotions_keys: list | dict | None = None):
    """
    ฟังก์ชันสำหรับนำเข้าข้อมูลจากไฟล์ CSV
    พารามิเตอร์:
        filename: ชื่อไฟล์ CSV
        text_column: ชื่อคอลัมน์ที่เก็บข้อความ
        emotions_column: ชื่อคอลัมน์ที่เก็บอารมณ์
        emotions_keys: พจนานุกรมหรือลิสต์สำหรับแปลงค่าอารมณ์ (ถ้ามี)
    """
    with open("./dataset/" + filename + ".csv", newline="", encoding="utf-8") as f:
        # อ่านไฟล์ CSV ด้วย pandas
        reader = pd.read_csv(f, encoding="utf-8")

        # วนลูปอ่านข้อมูลแต่ละแถว
        for index, row in reader.iterrows():
            texts.append(row[text_column])
            if emotions_keys:
                labels.append(emotions_keys[int(row[emotions_column])].lower())
            else:
                labels.append(row[emotions_column].lower())


# นำเข้าข้อมูลจากไฟล์ CSV ต่างๆ
importDataSetCsv("eng", "Comment", "Emotion")
importDataSetCsv("eng2", "text", "label")
importDataSetCsv("thai", "text", "emotion")

# สร้าง vectorizer สำหรับแปลงข้อความเป็นเวกเตอร์
# กำหนดให้ใช้ pythainlp tokenizer แทน default tokenizer
vectorizer = CountVectorizer(
    tokenizer=pythainlp.tokenize.word_tokenize, token_pattern=None)
# แปลงข้อความทั้งหมดเป็นเวกเตอร์
X = vectorizer.fit_transform(texts)

# สร้างและเทรนโมเดล Naive Bayes
# MultinomialNB เหมาะสำหรับการจำแนกข้อความที่มีลักษณะเป็น discrete features
model = MultinomialNB()
# เทรนโมเดลด้วยข้อมูลที่เตรียมไว้
model.fit(X, labels)


@app.get("/")
async def root(text: str):
    """
    API Endpoint สำหรับทำนายอารมณ์จากข้อความ
    รับพารามิเตอร์:
        text: ข้อความที่เข้ารหัสด้วย base64
    ส่งคืน:
        dict ที่มี key 'result' และค่าเป็นอารมณ์ที่ทำนายได้
    """
    # ถอดรหัส base64 เป็นข้อความปกติ
    data = base64.b64decode(text).decode("utf-8")
    # แปลงข้อความเป็นเวกเตอร์
    test_vector = vectorizer.transform([data])
    # ทำนายอารมณ์และส่งผลลัพธ์กลับ
    return {"result": model.predict(test_vector)[0]}

# ถ้ารันไฟล์นี้โดยตรง (ไม่ได้ import)
if __name__ == "__main__":
    # รัน FastAPI server ที่ localhost port 9959
    uvicorn.run(app, host="localhost", port=9959)