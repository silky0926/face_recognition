import shutil
import random
import glob
import os
from keras.utils.np_utils import to_categorical
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
import time


model = load_model('MyModel0606.h5')
cascade_path = "./haarcascades/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_path)

image = cv2.VideoCapture(1)#webカメラをキャプチャーする

while True:
	key = cv2.waitKey(1)
	if key != -1:
        	break

	ret, frame = image.read()
	image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	

	# 判定
	facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
	
	# 検出した場合
	if len(facerect) > 0:

		# 矩形作成
		# 第4引数で矩形の色を選択するがRGB形式で指定するとintegerしかダメと怒られます
		for rect in facerect:
			cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), 255, 2)
			img = frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
			img = cv2.resize(img, dsize=(250, 250))
			# cv2.imwrite("./detected/others" + str(time.time()) + "detected_face.jpg", img)
			pic = []
			pic.append(img)
			pic = np.array(pic)
			result = model.predict(pic)
			print(result)

		time.sleep(0.5)
	
	# 表示
	cv2.imshow('image', frame)

image.release()
cv2.destroyAllWindows()


# cascade_path = "./haarcascades/haarcascade_frontalface_alt.xml"
# path = "test/kato/0_-10.jpg"
# cascade = cv2.CascadeClassifier(cascade_path)
# img = cv2.imread(path)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

# # 検出した場合
# if len(facerect) > 0:

#     # 第4引数で矩形の色を選択するがRGB形式で指定するとintegerしかダメと怒られます
#     for rect in facerect:
#         img = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
#         img = cv2.resize(img, dsize=(250, 250))
#         # cv2.imwrite("./detected/others" + str(time.time()) + "detected_face.jpg", img)
#         pic = []
#         pic.append(img)
#         pic = np.array(pic)
#         result = model.predict(pic)
#         print(result)


# pic = []
# path = "test/kato/0_-10.jpg"
# pic.append(cv2.imread(path))
# pic = np.array(pic)
# result = model.predict(pic)

# print(result)