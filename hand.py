import os, cv2, operator
import numpy as np
from mathtools import dist
from othertools import rand_name
from collections import namedtuple
from recognizer import Recognizer
from args import *


class HandRecognizer(Recognizer):
    def raw_im_process(self):
        self.im = cv2.cvtColor(self.raw_im, cv2.COLOR_BGR2GRAY)
        ret, self.im = cv2.threshold(self.im, 150, 255, cv2.THRESH_BINARY)  # 不够白的都变黑
        # self._debug(self.im)

    def resume_order(self, arr, ykey, xkey):
        # 恢复现实顺序
        # 0 1 2
        # 3 4 5
        # 6 7 8
        arr.sort(key=ykey)  # 先按y轴排序
        temp = [sorted(arr[i:i + 3], key=xkey)
                for i in range(0, 9, 3)]  # 每三个一组按x轴排序
        return temp[0] + temp[1] + temp[2]

    def filter_contours(self):
        """
        筛选出九个矩形格子
        筛选条件：覆盖矩形不旋转过度，面积够大
        聚集、等大
        最终self.recs里面放的还是minAreaRect的结果
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
                    SUDOKU_WIDTH * LOW_THRESHOLD < w < SUDOKU_WIDTH * HIGH_THRESHOLD and
                    SUDOKU_HEIGHT * LOW_THRESHOLD < h < SUDOKU_HEIGHT * HIGH_THRESHOLD)


        self.recs = list(map(cv2.minAreaRect, self.contours))  # array([(cx, cy), (w, h), angle])

        self.recs = list(filter(legal, self.recs))  # 过滤过度旋转的矩形
        self.recs.sort(key=lambda it: it[1][0] * it[1][1], reverse=True)  # 过滤出面积前15大的矩形

        tmp = self.im.copy()
        for (cx, cy), (w, h), angle in self.recs:
            tmp = cv2.circle(tmp, (int(cx), int(cy)), 2, 100)
        self._debug(tmp)

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


    def choose_target_perspective(self):
        """透视变换"""
        RectCorner = namedtuple('RectCorner', ['lu', 'ru', 'ld', 'rd'])
        self.raw_corners = []
        for i, rec in enumerate(self.recs):
            box = [tuple(map(int, b)) for b in cv2.boxPoints(rec)]
            box.sort(key=lambda it: it[1])
            box = sorted(box[:2]) + sorted(box[2:])
            # print('test box orders', box)
            self.raw_corners.append(RectCorner(*box))  # raw_corners: 九个格子的四个角

        self.raw_corners = self.resume_order(self.raw_corners,
                                             ykey=lambda it: it.lu[1], xkey=lambda it: it.lu)

        sudoku = RectCorner(self.raw_corners[0].lu, self.raw_corners[2].ru,
                            self.raw_corners[6].ld, self.raw_corners[8].rd)
        sudoku_width = int(max(dist(sudoku.lu, sudoku.ru), dist(sudoku.ld, sudoku.rd)))
        sudoku_height = int(max(dist(sudoku.lu, sudoku.ld), dist(sudoku.ru, sudoku.rd)))

        tar1 = ((0, 0), (sudoku_width, 0), (0, sudoku_height), (sudoku_width, sudoku_height))
        H1 = cv2.getPerspectiveTransform(np.array(sudoku, dtype=np.float32),
                                        np.array(tar1, dtype=np.float32))
        self.im = cv2.warpPerspective(self.im, H1, (sudoku_width, sudoku_height))

        self._debug(self.im)

        tar2 = ((0, 1 / 2 * sudoku_height), (sudoku_width, 1 / 2 * sudoku_height),
                  (0, 1 / 2 * sudoku_height + sudoku_height),
                  (sudoku_width, 1 / 2 * sudoku_height + sudoku_height))
        H2 = cv2.getPerspectiveTransform(np.array(sudoku, dtype=np.float32),
                                         np.array(tar2, dtype=np.float32))
        self.raw_im = cv2.warpPerspective(self.raw_im, H2, (sudoku_width, int(3 / 2 * sudoku_height)))
        self.light = self.raw_im[0: int(1 / 2 * sudoku_height), :]

        # self._debug(self.light)

    def find_target_recs(self):  # 恢复现实顺序
        self.filter_contours()  # 以后程序的正确运行的前提：这九个轮廓就是目标轮廓
        self.choose_target_perspective()

        self.find_contours()
        self.recs = list(map(cv2.boundingRect, self.contours))
        self.recs.sort(key=lambda it: it[2] * it[3], reverse=True)
        self.recs = self.resume_order(self.recs[:9],
                                      ykey=lambda it: it[1], xkey=lambda it: it[0])


    def loop_process(self, func):
        self.init_knn()
        Recognizer.loop_process(self, func, pad=0.05)

    def init_knn(self):
        with np.load(MATERIAL_FILE) as db:
            train_data = db['train_data']
            train_label = db['train_label']

        self.knn = cv2.ml.KNearest_create()
        self.knn.train(train_data, cv2.ml.ROW_SAMPLE, train_label)

    def single_recognize(self, im):
        im = cv2.resize(im, TRAIN_SIZE)
        ret, im = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)

        self._debug(im)
        # cv2.imwrite(os.path.expanduser('~/Desktop/cutted/{}.jpg'.format(rand_name())), im)

        if 'cv_dight' in MATERIAL_FILE:
            im = 255 - im  # opencv自带的训练数据是黑底白字的

        im = im.reshape(-1, operator.mul(*TRAIN_SIZE)).astype(np.float32)
        retval, results, neign_resp, dists = self.knn.findNearest(im, 1)  # 目前看来k=1效果更好
        return int(results[0][0])

    def crop_light_image(self):
        return self.light


if __name__ == '__main__':
    HandRecognizer('test_im/real17498.jpg')
