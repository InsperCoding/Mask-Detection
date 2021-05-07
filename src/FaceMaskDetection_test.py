#!/usr/bin/python3.6

import cv2
import numpy as np
import argparse
from keras.models import load_model
import os
model = load_model("model2-010.model")
results = {0:'without mask', 1:'mask'}
GR_dict = {0:(0, 0, 255), 1:(0, 255, 0)}
rect_size = 4

def run(frame):
    frame_gray = cv2.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    rerect_size = cv2.resize(img, (img.shape[1] // rect_size, img.shape[0] // rect_size))
    faces = face_cascade.detectMultiScale(frame_gray)

    for (x,y,w,h) in faces:
        face_img = frame[y:y + h, x:x + w]
        rerect_sized = cv2.resize(face_img, (150, 150))
        normalized = rerect_sized/255.0
        reshaped = np.reshape(normalized, (1, 150, 150, 3))
        reshaped = np.vstack([reshaped])
        result = model.predict(reshaped)

        label = np.argmax(result, axis = 1)[0]
      
        cv2.rectangle(frame, (x, y), (x+w, y+h), GR_dict[label], 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), GR_dict[label], -1)
        cv2.putText(frame, results[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE', frame)


parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='/home/nicolas/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
face_cascade = cv2.CascadeClassifier()

if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)


cap = cv2.VideoCapture(0) 
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)


while True:
    (rval, img) = cap.read()
    img = cv2.flip(img, 1, 1)

    if not cap.isOpened():
        print('--(!)Error opening video capture')
        exit(0)

    run(img)
    
    if cv2.waitKey(10) == 27: 
        break

cap.release()
cv2.destroyAllWindows()
