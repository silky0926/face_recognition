#自分の顔画像収集用
# -*- coding: utf-8 -*-
import cv2
import time

# カスケード分類器名
cascade_path = "./haarcascades/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_path)

# WEBカメラの映像取得
# 内蔵カメラあり＋外付けの環境で外付けを使いたいときは(0)を(1)に変更
image = cv2.VideoCapture(1)

count = 500
while True:
	key = cv2.waitKey(1)
	if key != -1:
        	break

	ret, frame = image.read()
	# 速度向上のためグレースケール変換
	image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	

	# 判定
	facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
	
	# 検出した場合
	if len(facerect) > 0:
		print(facerect)
		for rect in facerect:
			cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), 255, 2)
			img = frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
			#顔部分のみ切り出して保存
			cv2.imwrite("./image/detected/kato/" +  str(count) + "detected_face.jpg", img)
			count += 1
		time.sleep(0.5)
	
	# 表示
	cv2.imshow('image', frame)

image.release()
cv2.destroyAllWindows()