# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 05:44:45 2018
对图片基于dlib库的人脸侦测，比opencv要准确
@author: Liujingtao
"""

import dlib
import cv2

imgpath='D:/IMG_0692.JPG'
img = cv2.imread(imgpath)
grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector() #使用默认的人类识别器模型
predictor = dlib.shape_predictor('D:/ai/trainingmodel/dlib/shape_predictor_68_face_landmarks.dat')

faces = detector(grayimg,1)
for face in faces:
    shape = predictor(img,face) # 寻找人脸的68个标定点
    for point in shape.parts(): # 遍历所有点，打印出其坐标，并圈出来
        point_postion = (point.x,point.y) 
        cv2.circle(img, point_postion, 2, (0, 255, 0), 1)
    cv2.imshow("image", img)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save image and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()