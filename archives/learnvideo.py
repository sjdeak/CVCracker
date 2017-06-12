import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while True:
    if cv2.waitKey(10) == ord('q'):
        break
    # ret, frame = cap.read()
    #
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1000) == ord('q'):
    #     break