import cv2
from trainer import VideoHandTrainer, init_dir
import numpy as np


init_dir()

MATERIAL_VIDEO = 'material.m4v'
cap = cv2.VideoCapture(MATERIAL_VIDEO)

while True:
    ret, frame = cap.read()  # todo 读取失败时的处理
    frame = frame[::-1,::-1]

    cv2.imshow('frame', frame)
    response = cv2.waitKey(100)

    if response == ord('q'):
        break
    if response == ord('t'):
        # todo 截出来的图还是有问题，检查透视变换的结果
        data = input()
        if data != 'pass':
            acNums = list(map(int, data))
            # print(acNums)
            VideoHandTrainer(frame, acNums)
