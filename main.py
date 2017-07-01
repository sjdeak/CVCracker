import cv2
import numpy as np
from collections import Counter
from hand import HandRecognizer
from light import LightRecognizer
from localize import Localizer
from args import VIDEO


CAL_ACCURACY_MODE = False
INVERT_IM = True


def read_frame(cap, invert=False):
    ret, frame = cap.read()
    if invert and ret:
        return ret, frame[::-1, ::-1]
    else:
        return ret, frame


def solve(light, hand):
    """根据识别结果做出最终判断"""
    cnter = Counter(hand)
    if not all(cnter[n] == 1 for n in light):
        return False

    return [hand.index(n) for n in light]


def mainloop():
    cap = cv2.VideoCapture(VIDEO)

    while True:
        ret, frame = read_frame(cap, invert=INVERT_IM)

        if not ret:
            break

        for i in range(3):
            try:
                state, light, hand = single_frame_test(frame, already_read=True)
                if not state:
                    ret, frame = read_invert_frame(cap)
                else:
                    print(solve(light, hand))
                    break
            except: pass
        else:
            cv2.waitKey(0)

if __name__ == '__main__':
    # single_frame_test('test_im/wrong3.jpg')
    mainloop()