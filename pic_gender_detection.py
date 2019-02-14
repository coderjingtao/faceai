# -*- coding: utf-8 -*-
"""
Created on Jul 30  2018
基于keras和tensorflow库，对图片中的人脸进行性别识别
@author: Liujingtao
"""
import cv2
from keras.models import load_model
import numpy as np

image = cv2.imread("D:/we3.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# OpenCV人脸识别分类器
face_classifier = cv2.CascadeClassifier('D:/ai/trainingmodel/haarcascades/haarcascade_frontalface_default.xml')
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(140, 140))

# keras性别分类器
gender_classifier = load_model('D:/ai/trainingmodel/gender_models/simple_CNN.81-0.96.hdf5')
gender_labels = {0:'female', 1:'male'}
color = (255, 255, 255)

# 遍历所有识别出的脸
for (x,y,w,h) in faces:
    face = image[(y - 60):(y + h + 60), (x - 30):(x + w + 30)]
    face = cv2.resize(face, (48, 48) )
    face = np.expand_dims(face, 0)
    face = face / 255.0
    gender_label_arg = np.argmax(gender_classifier.predict(face))
    gender = gender_labels[gender_label_arg]
    cv2.rectangle(image, (x, y), (x + h, y + w), color, 2)
    cv2.putText(image, gender, (x, y-3), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 1)

cv2.imshow("Gender_Detection",image)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save image and exit
    cv2.imwrite('gender.png',image)
    cv2.destroyAllWindows()




