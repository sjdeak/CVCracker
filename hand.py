import cv2
import numpy as np
from recognizer import Recognizer

class HandRecognizer(Recognizer):
    def crop_light_image(self):
        pass

    def raw_im_process(self):
        self.im = cv2.cvtColor(self.raw_im, cv2.COLOR_BGR2GRAY)
        ret, self.im = cv2.threshold(self.im, 150, 255, cv2.THRESH_BINARY)# 不够白的都变黑
        # cv2.imshow('hi', self.im)
        # cv2.waitKey(0)

    def find_target_recs(self):  # 获取矩形、筛选、恢复现实顺序
        def legal(rec):
            (cx, cy), (w, h), angle = rec

            if w == 0 or h == 0:
                return False

            directions = (0, 90, 180, 270)

            for d in directions:
                if (abs(abs(angle) - d) < ROTATE_BOUND and d in (90, 270)):
                    rec = (rec[0], (h, w), rec[2])
                    print(rec)
                    break

            # return abs(w / h - ASPECT_RATIO) < ASPECT_RATIO * PROPORTION
            return (any(abs(abs(angle) - d) < ROTATE_BOUND for d in directions) and
                    abs(rec[1][0] / rec[1][1] - ASPECT_RATIO) < ASPECT_RATIO * PROPORTION)


        ROTATE_BOUND = 10
        ASPECT_RATIO = 28 / 16
        PROPORTION = 1 / 5

        self.recs = list(map(cv2.minAreaRect, self.contours))

        self.recs = list(filter(legal, self.recs))  # 过滤过度旋转的矩形
        self.recs.sort(key=lambda it: it[1][0] * it[1][1], reverse=True)  # 根据面积排序
        self.recs = self.recs[:9]

        for i in range(9):
            rec = self.recs[i]
            box = cv2.boxPoints(rec)
            left_up = box[2]
            w, h = max(*rec[1]), min(*rec[1])
            self.recs[i] = tuple(map(int, (left_up[0], left_up[1], w, h)))


        # for rec in self.recs:
        #     box = cv2.boxPoints(rec)
        #     box = np.int0(box)
        #     cv2.drawContours(self.im, [box], 0, 100, 2)
        # self.debug()


if __name__ == '__main__':
    HandRecognizer('test_im/reality.jpg')
