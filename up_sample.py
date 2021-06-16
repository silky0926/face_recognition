import os
import cv2
import glob
from scipy import ndimage

"""
ディレクトリから画像を読み込んで回転、ぼかし、閾値処理をして保存する.
"""
SearchName = ['kato', 'kato_non_glass']
ImgSize=(250,250)

for name in SearchName:
    print("{}の写真を増やします。".format(name))
    in_dir = "./image/detected/"+name+"/*"
    out_dir = "./image/test_detected/"+name
    os.makedirs(out_dir, exist_ok=True)
    in_jpg=glob.glob(in_dir)
    
    for i in range(len(in_jpg)):
        #print(str(in_jpg[i]))
        img = cv2.imread(str(in_jpg[i]))
        # 回転
        # for ang in [-10,0,10]:
            # img_rot = ndimage.rotate(img,ang)
        img_rot = cv2.resize(img,ImgSize)
        fileName=os.path.join(out_dir,str(i)+".jpg")
        cv2.imwrite(str(fileName),img_rot)
            # # 閾値
            # img_thr = cv2.threshold(img_rot, 100, 255, cv2.THRESH_TOZERO)[1]
            # fileName=os.path.join(out_dir,str(i)+"_"+str(ang)+"thr.jpg")
            # cv2.imwrite(str(fileName),img_thr)
            # # ぼかし
            # img_filter = cv2.GaussianBlur(img_rot, (5, 5), 0)
            # fileName=os.path.join(out_dir,str(i)+"_"+str(ang)+"filter.jpg")
            # cv2.imwrite(str(fileName),img_filter)