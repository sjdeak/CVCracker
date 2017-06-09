import cv2
import numpy as np
from mathtools import dist
from collections import namedtuple
from recognizer import Recognizer

class HandRecognizer(Recognizer):
    def raw_im_process(self):
        self.im = cv2.cvtColor(self.raw_im, cv2.COLOR_BGR2GRAY)
        ret, self.im = cv2.threshold(self.im, 150, 255, cv2.THRESH_BINARY)# 不够白的都变黑

    def filter_contours(self):
        """
        筛选出九个矩形格子
        筛选条件：覆盖矩形不旋转过度，面积够大
        聚集、等大
        """
        def legal(rec):
            (cx, cy), (w, h), angle = rec

            if w == 0 or h == 0:
                return False

            directions = (0, 90, 180, 270)

            for d in directions:
                if (abs(abs(angle) - d) < ROTATE_BOUND and d in (90, 270)):
                    w, h = h, w  # todo 一开始试图修改rec造成遗忘，   规定：尽量不要用下标
                    break

            # return abs(w / h - ASPECT_RATIO) < ASPECT_RATIO * PROPORTION
            return (any(abs(abs(angle) - d) < ROTATE_BOUND for d in directions) and
                    abs(w / h - ASPECT_RATIO) < ASPECT_RATIO * PROPORTION and
                    SODOKU_WEIGHT * LOW_THRESHOLD < w < SODOKU_WEIGHT * HIGH_THRESHOLD and
                    SODOKU_HEIGHT * LOW_THRESHOLD < h < SODOKU_HEIGHT * HIGH_THRESHOLD)


        ROTATE_BOUND = 10  # 最多旋转几度
        ASPECT_RATIO = 28 / 16  # 标准宽高比
        PROPORTION = 1 / 5  # 和标准宽高比的最大差别界限

        LOW_THRESHOLD = 0.6
        HIGH_THRESHOLD = 1.4

        SODOKU_WEIGHT = 50  # 非常重要的两个参数，需要实际测量填写，足以决定成败
        SODOKU_HEIGHT = 28  # 当前宽高估计值

        self.recs = list(map(cv2.minAreaRect, self.contours))  # array([(cx, cy), (w, h), angle])

        self.recs = list(filter(legal, self.recs))  # 过滤过度旋转的矩形
        self.recs.sort(key=lambda it: it[1][0] * it[1][1], reverse=True)  # 过滤出面积前15大的矩形
        self.recs = self.recs[:15]

        if len(self.recs) < 9:
            pass  # todo 异常处理

        # 聚集筛选
        dist_sums = []
        for i in range(len(self.recs)):
            dist_sums.append(sum(dist(self.recs[i][0], self.recs[j][0])
                                 for j in range(len(self.recs)) if i != j))
        self.recs = [tp[0] for tp in sorted(zip(self.recs, dist_sums),
                                            key=lambda it: it[1])[:9]]

        # for rec in self.recs:
        #     box = cv2.boxPoints(rec)
        #     box = np.int0(box)
        #     cv2.drawContours(self.im, [box], 0, 100, 2)
        # self._debug(self.im)

    def choose_target_perspective(self):
        RectCorner = namedtuple('RectCorner', ['lu', 'ru', 'ld', 'rd'])
        self.raw_corners = []
        for i, rec in enumerate(self.recs):
            box = [tuple(map(int, b)) for b in cv2.boxPoints(rec)]
            box.sort(key=lambda it: it[1])
            box = sorted(box[:2]) + sorted(box[2:])
            # print('test box orders', box)
            self.raw_corners.append(RectCorner(*box))  # raw_corners: 九个格子的四个角

        # 恢复现实顺序
        # 0 1 2
        # 3 4 5
        # 6 7 8
        self.raw_corners.sort(key=lambda it: it.lu[1]) # 先按y轴排序
        temp = [sorted(self.raw_corners[i:i + 3], key=lambda it: it.lu)
                    for i in range(0, 9, 3)]  # 每三个一组按x轴排序
        self.raw_corners = temp[0] + temp[1] + temp[2]

        sudoku = RectCorner(self.raw_corners[0].lu, self.raw_corners[2].ru,
                            self.raw_corners[6].ld, self.raw_corners[8].rd)
        sudoku_width = int(max(dist(sudoku.lu, sudoku.ru), dist(sudoku.ld, sudoku.rd)))
        sudoku_height = int(max(dist(sudoku.lu, sudoku.ld), dist(sudoku.ru, sudoku.rd)))
        # print(sudoku, '|', sudoku_width, sudoku_height)
        target = ((0, 0), (sudoku_width, 0), (0, sudoku_height), (sudoku_width, sudoku_height))


        H = cv2.getPerspectiveTransform(np.array(sudoku, dtype=np.float32),
                                        np.array(target, dtype=np.float32))

        self.im = cv2.warpPerspective(self.im, H, (sudoku_width, sudoku_height))
        self._debug(self.im)

    def find_target_recs(self):  # 恢复现实顺序
        self.filter_contours()  # 以后程序的正确运行的前提：这九个轮廓就是目标轮廓
        self.choose_target_perspective()


        self.find_contours()
        self.recs = list(map(cv2.boundingRect, self.contours))
        self.recs.sort(key=lambda it: it[2] * it[3], reverse=True)
        self.recs = sorted(self.recs[:9], key=lambda it: it[1])
        temp = [sorted(self.recs[i:i + 3], key=lambda it: it[0])
                for i in range(0, 9, 3)]  # 每三个一组按x轴排序
        self.recs = temp[0] + temp[1] + temp[2]


    def loop_process(self, func):
        self.init_knn()
        Recognizer.loop_process(self, func, pad=0.05)

    def init_knn(self):
        train_im = cv2.imread('raw_train_materials/digits.png', 0)
        x = np.array([np.hsplit(row, 100) for row in np.vsplit(train_im, 50)])  # 切割开素材
        train_data = x[5:, :].reshape(-1, 400).astype(np.float32)  # (4500, 400)

        k = np.arange(1, 10)
        train_label = np.repeat(k, 500)[:, np.newaxis]  # (4500, 1)

        self.knn = cv2.ml.KNearest_create()
        self.knn.train(train_data, cv2.ml.ROW_SAMPLE, train_label)

    def single_recognize(self, im):
        # self._debug(im)
        im = cv2.resize(im, (20, 20))
        # todo 自己训练样本时记得去掉这行
        im = 255 - im  # 训练数据是黑底白字的

        im = im.reshape(-1, 400).astype(np.float32)
        retval, results, neign_resp, dists = self.knn.findNearest(im, 3)
        return int(results[0][0])

    def crop_light_image(self):
        rec0 = self.raw_corners[0].lu
        rec2 = self.raw_corners[2].lu
        w, h = (self.raw_corners[0].ru[0] - self.raw_corners[0].lu[0],
                self.raw_corners[0].ld[1] - self.raw_corners[0].lu[1])

        x0, y0 = map(int, (rec0[0] + 1 / 2 * w, rec0[1] - 3 / 2 * h))  # light 左上角
        x1, y1 = map(int, (rec2[0] + 1 / 2 * w, rec2[1] - 1 / 8 * h))  # light 右下角

        # self._debug(self.raw_im[y0:y1, x0:x1])

        y0, y1 = 0 if y0 < 0 else y0, 0 if y1 < 0 else y1  # todo 高度超出怎么办

        return self.raw_im[y0:y1, x0:x1]


if __name__ == '__main__':
    HandRecognizer('test_im/real17498.jpg')
