import cv2
import numpy as np
from collections import Counter
from hand import HandRecognizer
from light import LightRecognizer
from old import OldRecognizer
from localize import Localizer
from args import VIDEO, MODE
from serial import Serial
from exceptions import *


def cal_hand_targets(light, hand):
    """根据识别结果做出最终判断"""
    cnter = Counter(hand)
    if not all(cnter[n] == 1 for n in light):
        return False

    return [hand.index(n) for n in light]

def mainloop():
    ser = Serial('/dev/ttyUSB0')
    while True:
        signal = ser.read(10)
        if signal == b'AA':  # 定位
            pass

        elif signal == b'FF':  # 识别
            ims = []
            for i in range(3):
                ret, im = cv2.VideoCapture(0)
                if not ret:
                    break

                targets = []
                hander = HandRecognizer(im)
                try:
                    light_res = LightRecognizer(hander.light)
                    hand_res = hander.result
                except LightImError:  # 小符模式
                    targets = [OldRecognizer(im).result]
                except LightRecFail:
                    continue

                if not cal_hand_targets(hand_res, light_res):
                    continue

                for t in targets:  # 串口写回
                    # todo 大小符要分别标记
                    pass

                break




if __name__ == '__main__':
    # single_frame_test('test_im/wrong3.jpg')
    # mainloop()
    pass