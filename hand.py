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
                    # print(rec)
                    break

            # return abs(w / h - ASPECT_RATIO) < ASPECT_RATIO * PROPORTION
            return (any(abs(abs(angle) - d) < ROTATE_BOUND for d in directions) and
                    abs(rec[1][0] / rec[1][1] - ASPECT_RATIO) < ASPECT_RATIO * PROPORTION)


        ROTATE_BOUND = 10
        ASPECT_RATIO = 28 / 16
        PROPORTION = 1 / 5

        self.recs = list(map(cv2.minAreaRect, self.contours))  # array([(cx, cy), (w, h), angle])

        # todo 轮廓识别准确度优化，要做得更鲁棒才行
        self.recs = list(filter(legal, self.recs))  # 过滤过度旋转的矩形
        self.recs.sort(key=lambda it: it[1][0] * it[1][1], reverse=True)  # 根据面积排序
        self.recs = self.recs[:9]

        # for rec in self.recs:
        #     box = cv2.boxPoints(rec)
        #     box = np.int0(box)
        #     cv2.drawContours(self.im, [box], 0, 100, 2)
        # self.debugshow()

        # 必须保证这九个就是目标轮廓
        for i in range(9):
            rec = self.recs[i]
            w, h = max(*rec[1]), min(*rec[1])

            box = [tuple(b) for b in cv2.boxPoints(rec)]
            box.sort(key=lambda it: it[1])
            box = sorted(box[:2]) + box[2:]
            left_up = box[0]
            self.recs[i] = tuple(map(int, (left_up[0], left_up[1], w, h)))
            # print(self.recs[i])


        # 恢复现实顺序
        # 0 1 2
        # 3 4 5
        # 6 7 8
        self.recs.sort(key=lambda it: it[1])  # 先按y轴排序
        temp = [sorted(self.recs[i:i+3]) for i in range(0, 9, 3)]  # 每三个一组按x轴排序
        self.recs = temp[0] + temp[1] + temp[2]
        # print(self.recs)


    def loop_process(self, func):
        self.init_knn()
        Recognizer.loop_process(self, func)

    def init_knn(self):
        # todo pickle it
        train_im = cv2.imread('digits.png', 0)
        x = np.array([np.hsplit(row, 100) for row in np.vsplit(train_im, 50)])  # 切割开素材
        train_data = x[5:, :].reshape(-1, 400).astype(np.float32)  # (4500, 400)

        k = np.arange(1, 10)
        train_label = np.repeat(k, 500)[:, np.newaxis]  # (4500, 1)

        self.knn = cv2.ml.KNearest_create()
        self.knn.train(train_data, cv2.ml.ROW_SAMPLE, train_label)

    def single_recognize(self, im):
        im = cv2.resize(im, (20, 20))
        # todo 自己训练样本时记得去掉这行
        im = 255 - im  # 训练数据是黑底白字的

        # cv2.imshow('hi', im)
        # cv2.waitKey(0)

        im = im.reshape(-1, 400).astype(np.float32)
        retval, results, neign_resp, dists = self.knn.findNearest(im, 3)
        print(results)



if __name__ == '__main__':
    HandRecognizer('test_im/real3.jpg')
