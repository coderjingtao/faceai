# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 06:15:52 2018
对视频，基于dlib库的人脸侦测，检测效果好于opencv,但速度教慢
@author: Liujingtao
"""

import dlib
import cv2

detector = dlib.get_frontal_face_detector()  #使用默认的人类识别器模型

def faceRecognize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    for face in faces:
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imshow("image", img)
        
camera = cv2.VideoCapture(1)
while camera.isOpened():
    ret, frame = camera.read()# 逐帧读取
    if ret == True:
        faceRecognize(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()