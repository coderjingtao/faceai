# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 18:32:33 2018
对图片基于opencv库的人脸侦测
@author: Liujingtao
"""
import cv2

imgpath='D:/test.jpg'
img = cv2.imread(imgpath)
grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# OpenCV人脸识别分类器
classifier = cv2.CascadeClassifier('D:/ai/trainingmodel/haarcascades/haarcascade_frontalface_default.xml')
color = (0, 255, 0)  # 定义绘制颜色
# 调用识别人脸
faces = classifier.detectMultiScale(grayimg, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
# 大于0则检测出人脸
if len(faces) > 0 :
    for face in faces: #框出每一张人脸
       x, y, w, h = face
       #face
       cv2.rectangle(img,(x,y),(x+h,y+w),color,2)
       #left
       cv2.circle(img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
       #right eye
       cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
       #mouse
       cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4),(x + 5 * w // 8, y + 7 * h // 8), color)
       
cv2.imshow("face recognition",img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save image and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()

