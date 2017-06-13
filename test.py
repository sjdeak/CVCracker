import cv2
import numpy as np
from collections import Counter
from hand import HandRecognizer
from light import LightRecognizer
from localize import Localizer
from args import VIDEO

# hand手工训练
# light优化
# weird num 冲突处理
# todo 串口

# 半自动化：操作手处理后两位
# 各数字黑色像素点数统计，用于猜测数字
# 自定义异常
# 文档、注释
CAL_ACCURACY_MODE = False

def single_frame_test(im, already_read=False):
    hr = HandRecognizer(im, already_read=already_read)
    light_im = hr.crop_light_image()
    lr = LightRecognizer(light_im, already_read=True)

    if 'X' not in lr.result and len(lr.result) == 5:
        print('-----------\n', lr.result)
        print(np.array(hr.result).reshape(3, 3))
        return True, lr.result, hr.result
    else:
        return False, 0, 0


def read_invert_frame(cap):
    ret, frame = cap.read()
    return ret, frame[::-1, ::-1]


def solve(light, hand):
    """根据识别结果做出最终判断"""
    cnter = Counter(hand)
    if not all(cnter[n] == 1 for n in light):
        return False

    return [hand.index(n) for n in light]


def mainloop():
    cap = cv2.VideoCapture(VIDEO)

    wrong_light, wrong_hand = 0, 0
    sum_light, sum_hand = 0, 0

    while True:
        ret, frame = read_invert_frame(cap)

        if not ret:
            if CAL_ACCURACY_MODE:
                print('accuracy:')
                print('light: {}'.format(1 - wrong_light / sum_light))
                print('hand: {}'.format(1 - wrong_hand / sum_hand))

            break

        cv2.imshow('frame', frame)
        response = cv2.waitKey(100)
        if response == ord('q'):
            break

        if response == ord('r'):
            print(Localizer(frame, already_read=True).move_value())

        if response == ord('t'):
            # 图像抖动 ok
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
                print('Error')

            if CAL_ACCURACY_MODE:
                data = input()
                if data != 'pass':
                    wl, wh = map(int, data.split())
                    wrong_light += wl
                    wrong_hand += wh
                    sum_light += 5
                    sum_hand += 9
            else:
                cv2.waitKey(0)

if __name__ == '__main__':
    # single_frame_test('test_im/wrong3.jpg')
    mainloop()