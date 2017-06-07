import cv2
import numpy as np


cap = cv2.VideoCapture('../rune.m4v')

while True:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1000) == ord('q'):
        break