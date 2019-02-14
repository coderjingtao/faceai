# -*- coding: utf-8 -*-
"""
Created on Jul 30  2018
基于keras库和tensorflow库，对图片中的人脸进行情绪识别
@author: Liujingtao
"""

import cv2
from keras.models import load_model
import numpy as np

# loading emotional model
emotion_classifier = load_model('D:/ai/trainingmodel/emotion_models/simple_CNN.985-0.66.hdf5')
emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'low'}
#emotion_labels = {0: '生气',1: '厌恶',2: '恐惧',3: '开心',4: '难过',5: '惊喜',6: '平静 neutral'}

# loading
image = cv2.imread("D:/emotions.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_classifier = cv2.CascadeClassifier('D:/ai/trainingmodel/haarcascades/haarcascade_frontalface_default.xml')
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))
color = (255, 255, 255)

# 遍历所有识别出的脸
for (x,y,w,h) in faces:
    gray_face = gray[(y):(y + h), (x):(x + w)]
    gray_face = cv2.resize(gray_face, (48, 48))
    gray_face = gray_face / 255.0
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion = emotion_labels[emotion_label_arg]
    cv2.rectangle(image, (x + 10, y + 10), (x + h - 10, y + w - 10),color, 2)
    cv2.putText(image, emotion, (x, y-3), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 1)

cv2.imshow("Emotion_Detection",image)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save image and exit
    cv2.imwrite('emotion.png',image)
    cv2.destroyAllWindows()