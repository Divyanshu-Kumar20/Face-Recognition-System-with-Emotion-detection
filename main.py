import cv2
import face_recognition
import numpy as np
import os
from deepface import DeepFace
from datetime import datetime
import sqlite3
import smtplib

# ========================= DATABASE SETUP =========================

conn = sqlite3.connect('emotion_data.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS emotions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    emotion TEXT,
    date TEXT,
    time TEXT
)
''')

conn.commit()

# ========================= ATTENDANCE SYSTEM =========================

def markAttendance(name, emotion):

    if not os.path.exists("attendance.csv"):
        with open("attendance.csv", "w") as f:
            f.write("Name,Date,Time,Emotion\n")

    with open("attendance.csv", "a") as f:
        now = datetime.now()

        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        f.write(f"{name},{date},{time},{emotion}\n")

# ========================= DATABASE SAVE =========================

def saveEmotion(name, emotion):

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    cursor.execute(
        "INSERT INTO emotions (name, emotion, date, time) VALUES (?, ?, ?, ?)",
        (name, emotion, date, time)
    )

    conn.commit()

# ========================= EMAIL ALERT =========================

def sendAlert(name, emotion):

    sender_email = "yourgmail@gmail.com"
    sender_password = "your_app_password"

    receiver_email = "receiver@gmail.com"

    subject = "Emotion Alert"
    body = f"Alert! {name} detected with emotion: {emotion}"

    message = f"Subject: {subject}\n\n{body}"

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()

        server.login(sender_email, sender_password)

        server.sendmail(sender_email, receiver_email, message)

        server.quit()

        print("Alert Email Sent")

    except Exception as e:
        print("Email Error:", e)

# ========================= LOAD DATASET =========================

path = 'dataset'

if not os.path.exists(path):
    os.makedirs(path)
    print(f"Please add images inside '{path}' folder and restart.")
    exit()

images = []
classNames = []

myList = os.listdir(path)

print("Loading Images...")

for cl in myList:

    curImg = cv2.imread(f'{path}/{cl}')

    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    else:
        print(f"Skipping {cl}")

# ========================= ENCODING =========================

def findEncodings(images):

    encodeList = []

    for img in images:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        encodes = face_recognition.face_encodings(img)

        if len(encodes) > 0:
            encodeList.append(encodes[0])

        else:
            print("Warning: No face found in one image")

    return encodeList

print("Encoding Faces...")

encodeListKnown = findEncodings(images)

print("Encoding Complete!")

# ========================= WEBCAM =========================

cap = cv2.VideoCapture(0)

print("Starting Camera... Press 'Q' to exit")

already_marked = set()

while True:

    success, img = cap.read()

    if not success:
        break

    # Resize image for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)

    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces
    facesCurFrame = face_recognition.face_locations(imgS)

    encodesCurFrame = face_recognition.face_encodings(
        imgS,
        facesCurFrame
    )

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

        matches = face_recognition.compare_faces(
            encodeListKnown,
            encodeFace
        )

        faceDis = face_recognition.face_distance(
            encodeListKnown,
            encodeFace
        )

        name = "UNKNOWN"

        if len(faceDis) > 0:

            matchIndex = np.argmin(faceDis)

            # Better accuracy threshold
            if matches[matchIndex] and faceDis[matchIndex] < 0.50:
                name = classNames[matchIndex].upper()

        # Scale back up face locations
        y1, x2, y2, x1 = faceLoc

        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        # ========================= EMOTION DETECTION =========================

        try:

            face_img = img[y1:y2, x1:x2]

            if face_img.size != 0:

                analysis = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    enforce_detection=False
                )

                emotion = analysis[0]['dominant_emotion']

            else:
                emotion = "N/A"

        except:
            emotion = "N/A"

        # ========================= SAVE DATA =========================

        unique_entry = f"{name}-{emotion}"

        if unique_entry not in already_marked:

            markAttendance(name, emotion)

            saveEmotion(name, emotion)

            already_marked.add(unique_entry)

        # ========================= EMAIL ALERT =========================

        if emotion == "angry":
            sendAlert(name, emotion)

        # ========================= LABEL =========================

        label = f"{name} - {emotion}"

        # ========================= DRAW =========================

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.rectangle(
            img,
            (x1, y2 - 40),
            (x2, y2),
            (0, 255, 0),
            cv2.FILLED
        )

        cv2.putText(
            img,
            label,
            (x1 + 6, y2 - 10),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow("Face Recognition + Emotion Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ========================= CLEANUP =========================

cap.release()

cv2.destroyAllWindows()

conn.close()