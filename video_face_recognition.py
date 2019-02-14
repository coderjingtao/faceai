# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:33:18 2018
对视频，基于face_recognition库的人脸识别，并支持视频录制
@author: Liujingtao
"""

import cv2
import face_recognition
import os
import time

captureVideo = False

if captureVideo == True:
    # Define the codec and create VideoWriter object
    videoFormat = cv2.VideoWriter_fourcc(*'XVID')
    filename = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
    out = cv2.VideoWriter('D:/ai/result/'+filename+'.avi',videoFormat, 30.0, (640,480))

# step1: 训练模型并编码
model_path = "D:/ai/trainingmodel/my" #模型数据图片目录
total_image_name = []
total_face_encoding = []

for file_name in os.listdir(model_path):
    image_path = model_path+"/"+file_name
    print(image_path)
    total_face_encoding.append(face_recognition.face_encodings(face_recognition.load_image_file(image_path))[0])
    file_name = file_name[:(len(file_name)-4)] #截取图片名称，去掉.jpg,(图片名为人物名)
    total_image_name.append(file_name)



# step2: 识别图像中的人物是否和模型中匹配
camera = cv2.VideoCapture(0)
while camera.isOpened():
    ret, frame = camera.read()
    #扫描每一帧中的所有脸并编码
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame,face_locations)
    #遍历每一张脸
    for (top, right, bottom, left), face_encoding in zip(face_locations,face_encodings):
        for i,v in enumerate(total_face_encoding):
            match = face_recognition.compare_faces([v],face_encoding,tolerance=0.5)
            name = "Unknown"
            if match[0]:
                name = total_image_name[i]
                break
        #给脸画一个框
        cv2.rectangle(frame, (left,top), (right,bottom), (0,0,255), 2)
        #在框下显示一个label
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    #显示最终图像
    cv2.imshow('face recognition', frame)
    if captureVideo == True:
        out.write(frame) #save video to disk
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#释放资源
camera.release()
out.release()
cv2.destroyAllWindows()
