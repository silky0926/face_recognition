import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

import cv2
from keras.models import load_model

#スプレッドシートを扱うためのライブラリ
import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials

def connect_gspread(jsonf, key):
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    #認証情報の設定
    credentials = ServiceAccountCredentials.from_json_keyfile_name(jsonf, scope)
    gc = gspread.authorize(credentials)
    #スプレッドシートキーを用いてsheet1にアクセス
    SPREADSHEET_KEY = key
    worksheet = gc.open_by_key(SPREADSHEET_KEY).sheet1
    return worksheet

def write_sheet():
	#Spread Sheets上の値を取得
	values = ws.get_all_values()
	row_num = len(values)
	col_num = len(values[0])

	#Spread Sheets上の値を更新
	date_check = values[row_num-1][0]
	days = datetime.date.today()
	now = datetime.datetime.now()
	d_time = str(now.hour) + ':' + str(now.minute)

	#日付の書き込み
	if(date_check != str(days)):
		ws.update_cell(row_num+1,1,str(days))
		ws.update_cell(row_num+1,2,str(d_time))
	else:
		ws.update_cell(row_num,1,str(days))
		ws.update_cell(row_num,3,str(d_time))
	
	print('detect:' + str(now))


def face_detector(model, cascade):
	#webカメラをキャプチャーする
	image = cv2.VideoCapture(1)
	count = 0

	while True:
		key = cv2.waitKey(1)
		if key != -1:
				break

		ret, frame = image.read()
		image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	

		#顔検出
		facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

		#検出した場合
		if len(facerect) > 0:
			for rect in facerect:
				img = frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

				#顔部分に矩形を表示
				cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), 255, 2)
				
				#予測用に画像の形式を変更
				img = cv2.resize(img, dsize=(250, 250))
				r,g,b = cv2.split(img)
				img = cv2.merge([r,g,b])
				pic = []
				pic.append(img)
				pic = np.array(pic)
				pic = pic/255

				#予測
				result = model.predict(pic)
				print(result[0][0])
				if(float(result[0][0]) > 0.8):
					count += 1
				else:
					count = 0
				
				#3フレーム連続で認識した場合のみ在室とみなす
				if(count >= 3):
					write_sheet()

		#1秒ごとに顔認識
		time.sleep(1)
		
		# 表示
		cv2.imshow('image', frame)

	image.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':#直接yobareru.pyを実行した時だけ、def test()を実行する
	#共有設定したスプレッドシートキーを指定
	jsonf = ''
	spread_sheet_key = ''
	ws = connect_gspread(jsonf, spread_sheet_key)

	#検知器の起動
	model = load_model('MyModel0617.h5')
	cascade_path = "./haarcascades/haarcascade_frontalface_alt.xml"
	cascade = cv2.CascadeClassifier(cascade_path)
	face_detector(model, cascade)