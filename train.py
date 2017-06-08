import cv2
from trainer import VideoHandTrainer, init_dir
import numpy as np


init_dir()

MATERIAL_VIDEO = 'raw_train_materials/material.m4v'
cap = cv2.VideoCapture(MATERIAL_VIDEO)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = frame[::-1,::-1]

    cv2.imshow('frame', frame)
    response = cv2.waitKey(100)
    if response == ord('q'):
        break
    if response == ord('t'):
        print('start train single frame')
        data = '123456789'
        # data = input()
        if data != 'pass':
            acNums = list(map(int, data))
            VideoHandTrainer(frame, acNums)
