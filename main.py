# นำเข้าไลบรารีที่จำเป็น
import pythainlp  # สำหรับประมวลผลและตัดคำภาษาไทย
import pandas as pd  # สำหรับจัดการข้อมูลในรูปแบบ DataFrame
import base64  # สำหรับเข้ารหัสและถอดรหัสข้อความ
from fastapi import FastAPI  # สำหรับสร้าง Web API
from fastapi.middleware.cors import CORSMiddleware  # สำหรับจัดการ CORS policy
from sklearn.feature_extraction.text import CountVectorizer # สำหรับแปลงข้อความเป็นเวกเตอร์
from sklearn.naive_bayes import MultinomialNB # โมเดล Naive Bayes สำหรับการจำแนกข้อความ
import uvicorn  # สำหรับรันเซิร์ฟเวอร์ FastAPI

# สร้าง FastAPI application instance
app = FastAPI()

# กำหนด origins ที่อนุญาตให้เข้าถึง API (ใช้สำหรับ CORS policy)
origins = [
    "http://localhost",
    "http://localhost:3000",
]

# กำหนด middleware เพื่อจัดการ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# สร้างลิสต์สำหรับเก็บข้อความและอารมณ์ที่ใช้ในการเทรนโมเดล
texts = []
labels = []


def importDataSetCsv(filename: str, text_column: str, emotions_column: str, emotions_keys: list | dict | None = None):
    """
    ฟังก์ชันสำหรับนำเข้าข้อมูลจากไฟล์ CSV เพื่อนำไปใช้ในการฝึกโมเดล
    พารามิเตอร์:
        filename: ชื่อไฟล์ CSV (ไม่ต้องใส่นามสกุล .csv)
        text_column: ชื่อคอลัมน์ที่เก็บข้อความ
        emotions_column: ชื่อคอลัมน์ที่เก็บอารมณ์ของข้อความ
        emotions_keys: พจนานุกรมหรือลิสต์สำหรับแปลงค่าอารมณ์เป็นข้อความ (ถ้ามี)
    """
    with open("./dataset/" + filename + ".csv", newline="", encoding="utf-8") as f:
        # อ่านไฟล์ CSV เป็น DataFrame ของ pandas
        reader = pd.read_csv(f, encoding="utf-8")

        # วนลูปอ่านข้อมูลแต่ละแถว และเพิ่มลงในลิสต์ texts และ labels
        for index, row in reader.iterrows():
            texts.append(row[text_column])  # เพิ่มข้อความลงในลิสต์
            if emotions_keys:
                # แปลงค่าอารมณ์ (ถ้ามี keys)
                labels.append(emotions_keys[int(row[emotions_column])].lower())
            else:
                # เพิ่มค่าอารมณ์ลงในลิสต์
                labels.append(row[emotions_column].lower())


# นำเข้าข้อมูลจากไฟล์ CSV ที่ใช้สำหรับฝึกโมเดล
importDataSetCsv("eng", "Comment", "Emotion")  # ข้อมูลภาษาอังกฤษชุดที่ 1
importDataSetCsv("eng2", "text", "label")  # ข้อมูลภาษาอังกฤษชุดที่ 2
importDataSetCsv("thai", "text", "emotion")  # ข้อมูลภาษาไทย

# สร้าง vectorizer สำหรับแปลงข้อความเป็นเวกเตอร์
# ใช้ pythainlp tokenizer แทน default tokenizer ของ CountVectorizer
vectorizer = CountVectorizer(
    tokenizer=pythainlp.tokenize.word_tokenize, token_pattern=None
)

# แปลงข้อความทั้งหมดเป็นเวกเตอร์
X = vectorizer.fit_transform(texts)

# สร้างโมเดล Naive Bayes (MultinomialNB) สำหรับการจำแนกอารมณ์ของข้อความ
model = MultinomialNB()

# เทรนโมเดลด้วยข้อมูลที่เตรียมไว้
model.fit(X, labels)


@app.get("/")
async def root(text: str):
    """
    API Endpoint สำหรับทำนายอารมณ์ของข้อความที่ได้รับ
    พารามิเตอร์:
        text: ข้อความที่เข้ารหัสด้วย base64
    ส่งคืน:
        JSON ที่มี key 'result' และค่าคืออารมณ์ที่ทำนายได้
    """
    # ถอดรหัสข้อความจาก base64
    data = base64.b64decode(text).decode("utf-8")

    # แปลงข้อความที่รับเข้ามาเป็นเวกเตอร์
    test_vector = vectorizer.transform([data])

    # ใช้โมเดล Naive Bayes ทำนายอารมณ์ของข้อความ
    predicted_emotion = model.predict(test_vector)[0]

    # ส่งผลลัพธ์กลับเป็น JSON
    return {"result": predicted_emotion}

# ถ้ารันไฟล์นี้โดยตรง (ไม่ได้ถูก import ไปใช้ที่อื่น)
if __name__ == "__main__":
    # รัน FastAPI server ที่ localhost port 9959
    uvicorn.run(app, host="localhost", port=9959)
