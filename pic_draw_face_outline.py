#coding=utf-8
#Draw the outline of a face
#对图片基于face_recognition库的人脸轮廓描绘
import face_recognition
from PIL import Image, ImageDraw

# load one picture with a face into the array of numpy
image = face_recognition.load_image_file('D:/jingtao.jpg')

# search for all the characteristics of the face
face_landmark_list = face_recognition.face_landmarks(image)

# iterate facial features and draw them
for face_landmark in face_landmark_list:
    facial_features = [
        'chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip',
        'left_eye', 'right_eye', 'top_lip', 'bottom_lip'
    ]
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    for feature in facial_features:
        d.line(face_landmark[feature],fill=(255,255,255),width=3)
    pil_image.show()
