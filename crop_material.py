"""切割从视频中切割并分类九宫格图片，为之后的训练准备素材"""
import os, cv2, shutil
from trainer import VideoHandTrainer
import numpy as np


MATERIAL_VIDEO = 'raw_train_materials/material.m4v'

def init_dir():
    shutil.rmtree('train_hand')
    os.mkdir('train_hand')
    for i in range(1, 10):
        os.mkdir('train_hand/{}'.format(i))


init_dir()

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
        print('请输入九宫格块的实际数字')
        # data = '123456789'
        data = input()
        if data != 'pass':
            acNums = list(map(int, data))
            VideoHandTrainer(frame, acNums)
