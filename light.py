import cv2
import numpy as np
from recognizer import Recognizer

# todo 去掉白光
RED = (np.array([0, 0, 210]), np.array([255, 255, 255]))


class LightRecognizer(Recognizer):
    DIGIT = {
        (1, 0, 1, 1, 0, 1, 1): 2,
        (1, 0, 0, 1, 1, 1, 1): 3,
        (0, 1, 0, 0, 1, 1, 1): 4,
        (1, 1, 0, 1, 1, 0, 1): 5,
        (1, 1, 1, 1, 1, 0, 1): 6,
        (1, 0, 0, 0, 1, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 0, 1, 1, 1, 1): 9
    }

    def raw_im_process(self):
        self.im = cv2.inRange(self.raw_im, *RED)

    def find_target_recs(self):
        # todo 检查五个轮廓距离是否够近
        self.recs = list(map(cv2.boundingRect, self.contours))
        self.recs.sort(key=lambda it: it[2] * it[3], reverse=True)  # 根据面积排序
        self.recs = self.recs[:5]
        self.recs.sort()  # 根据左右位置排序

    def check_line(self, pos, direction, im):
        h, w = im.shape
        dis = im.shape[1] // 7  # 宽度的1/4

        for i in range(-dis, dis):
            p = list(pos)
            p[direction] += i

            nr, nc = p
            if not (nr in range(h) and nc in range(w)):
                continue
            # todo 不能1个白点就判1
            if im[nr, nc] == 255:
                return 1
        else:
            return 0

    def single_recognize(self, im):
        h, w = im.shape

        if h / w > 2.5:
            return 1

        ch, cw = (h // 2, w // 2)
        offset = w // 8

        a, d = (0, cw), (h, cw)
        b, f = (ch // 2, offset), (ch // 2, w)
        c, e = (ch + ch // 2, 0), (ch + ch // 2, w - offset)
        g = (ch, cw)

        ver, hor = 0, 1
        checkpoints = [a, b, c, d, e, f, g]
        check_directions = [ver, hor, hor, ver, hor, hor, ver]

        res = tuple(self.check_line(*it, im)
                       for it in zip(checkpoints, check_directions))

        try:
            return LightRecognizer.DIGIT[res]
        except KeyError:
            print('Error', res)
            return 'Error!'


if __name__ == '__main__':
    # print(LightRecognizer('test_im/98564.jpg').result)
    print(LightRecognizer('material_light/34921.jpg').result)
