import cv2
import numpy as np
from hand import HandRecognizer
from light import LightRecognizer


FILE_NAME = 'test_im/real3.jpg'

hr = HandRecognizer(FILE_NAME)
light_im = hr.crop_light_image()
cv2.imwrite('light.jpg', light_im)

lr = LightRecognizer('light.jpg')


from tkinter import *

root = Tk()
root.geometry('300x300')


Label(text=''.join(str(n) for n in lr.result)).grid(row=0, columnspan=3, sticky=NSEW)
hand_result = np.array(hr.result, dtype=np.uint8).reshape(3,3)

for i in range(3):
    for j in range(3):
        n = hand_result[i][j]
        Label(text=str(n)).grid(row=i+1, column=j, sticky=NSEW)

mainloop()