# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:33:18 2018
对视频，基于opencv库的人脸侦测，检测效果不如dlib,但速度教快
@author: Liujingtao
"""

import cv2
import time


captureVideo = True

if captureVideo == True:
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    filename = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
    out = cv2.VideoWriter('D:/ai/result/'+filename+'.avi',fourcc, 20.0, (640,480))

def recognizeFace(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    classifier = cv2.CascadeClassifier('D:/ai/trainingmodel/haarcascades/haarcascade_frontalface_default.xml')
    faces = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
    if len(faces) > 0 :
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)  # 框出人脸
    cv2.imshow("Image", img)
    if captureVideo == True:
        out.write(img) #save video to disk

cap = cv2.VideoCapture(0)

while cap.isOpened():  
    ret, frame = cap.read()# 逐帧读取
    if ret == True:
        recognizeFace(frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()