# นำเข้าไลบรารีที่จำเป็น
import pythainlp
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

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
    with open("./dataset/csv/" + filename + ".csv", newline="", encoding="utf-8") as f:
        # อ่านไฟล์ CSV ด้วย pandas
        reader = pd.read_csv(f, encoding="utf-8")

        # วนลูปอ่านข้อมูลแต่ละแถว
        for index, row in reader.iterrows():
            texts.append(row[text_column])
            if emotions_keys:
                labels.append(emotions_keys[int(row[emotions_column])])
            else:
                labels.append(row[emotions_column].lower())


# นำเข้าข้อมูลจากไฟล์ CSV ต่างๆ
importDataSetCsv("training", "text", "label", {
                 0: 'Sadnessness', 1: 'happy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'})

importDataSetCsv("Emotion_classify_Data", "Comment", "Emotion")

importDataSetCsv("thai_emotion_dataset_large", "text", "emotion")

# สร้าง vectorizer สำหรับแปลงข้อความเป็นเวกเตอร์
vectorizer = CountVectorizer(
    tokenizer=pythainlp.tokenize.word_tokenize, token_pattern=None)
X = vectorizer.fit_transform(texts)

# สร้างและเทรนโมเดล Naive Bayes
model = MultinomialNB()
model.fit(X, labels)

# รับข้อความจากผู้ใช้และทำนายอารมณ์
test_text = str(input("Enter text: "))
test_vector = vectorizer.transform([test_text])

print("Predicted emotion ::", model.predict(test_vector)[0])
