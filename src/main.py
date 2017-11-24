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
            ret, im = cv2.VideoCapture(0)
            if not ret:
                break

            try:
                mv_value = Localizer(im).move_value
            except:
                print('定位出错')
            else:
                pass  # 写回

        elif signal == b'FF':  # 识别
            for i in range(3):
                ret, im = cv2.VideoCapture(0)
                if not ret:
                    break

                targets = []
                hander = HandRecognizer(im)
                rune_type = 'new'
                try:
                    light_res = LightRecognizer(hander.light).result
                    hand_res = hander.result
                except LightImError:
                    rune_type = 'old'
                    targets = [OldRecognizer(im).result]
                    print('小符模式', targets)
                except LightRecFail as X:
                    print('第{}个七段管识别出错: {}', X.id, X.info)
                    continue


                if rune_type == 'new':
                    targets = cal_hand_targets(hand_res, light_res)
                    if not targets:
                        continue
                    for t in targets:  # 串口写回
                        pass

                if rune_type == 'old':
                    pass

                break


if __name__ == '__main__':
    # single_frame_test('test_im/wrong3.jpg')
    # mainloop()
    pass