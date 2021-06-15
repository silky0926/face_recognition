import cv2
import numpy as np
from keras.models import load_model
from detect_push import detectFace, pushToLine, pushToSlack
import time

SAVE_PATH = "./input/face.jpg"
model = load_model('faces.h5')
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)#webカメラをキャプチャーする
while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)#反転
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#グレースケール化
    cv2.imshow("img",img)
    faces = face_cascade.detectMultiScale(gray, 1.9, 5)
    if len(face)>0:      
        for rect in face:
            cv2.imwrite(SAVE_PATH, img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]])
            print("recognize face !!")

            member = detectFace(SAVE_PATH, model)#上で保存した画像を読み込み、モデルでの予測結果をmemberに代入
            pushToLine(member)#予測結果を受け取り、LINEbotから送信
            pushToSlack(member)#予測結果を受け取り、slackbotから送信

            time.sleep(10)
    if cv2.waitKey(30) == 27:#Escキーを押すとbreak
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()
