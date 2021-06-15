#顔写真収集用
import cv2
import glob
import numpy as np
import os

cascade_path = "./haarcascades/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascade_path)#顔検出器を生成

files = glob.glob("./detected/others/*")
save_path = "./detected/others2/"

face_detect_count = 0
# for fname in files:  
#     # 顔検知に成功した数(デフォルトで0を指定)
#     bgr = cv2.imread(fname, cv2.IMREAD_COLOR)
#     image_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

#     img = cv2.imread(fname)
#     face = faceCascade.detectMultiScale(image_gray, 1.1, 3)
#     if len(face) > 0:
#         for rect in face:
#             # 顔認識部分を赤線で囲み保存(今はこの部分は必要ない)
#             # cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0,255), thickness=1)
#             # cv2.imwrite('detected.jpg', img)
#             x = rect[0]
#             y = rect[1]
#             w = rect[2]
#             h = rect[3]
#             cv2.imwrite(save_path + 'cutted' + str(face_detect_count) + '.jpg',img[y:y+h,  x:x+w])
#             face_detect_count = face_detect_count + 1
#             print("now")

# print("顔画像の切り取り作業、正常に動作しました。")

face_detect_count = 0
for fname in files:
        bgr = cv2.imread(fname)
        bgr = cv2.resize(bgr, dsize=(250,250))
        cv2.imwrite(save_path + 'cutted' + str(face_detect_count) + '.jpg', bgr)
        face_detect_count += 1