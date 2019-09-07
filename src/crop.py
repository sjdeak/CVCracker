import os, sys, cv2
import numpy as np
from detect import Detector
from utility import dist, randName
from collections import namedtuple
from args import *


DEBUG = True
SAVE = False


class Cropper:
    """单一目标切割成若干份"""
    def __init__(self):
        self.raw_im = None
        self.threshold_im = None
        self.results = None
        self.contours = None
        self.rec_contours = None
    
    def getRawIm(self, im):
        self.raw_im = im
        
    def imshow(self, im, msg='None'):
        """调试函数，显示im"""
        cv2.imshow(msg, im)
        cv2.waitKey(0)
    
    def saveCropResults(self, results=None, fnames=[], path=os.path.expanduser('~/Desktop')):
        """
        :param results: [图片1, 图片2, ...]
        :param fnames: [fname1, fname2, ...]
        :param path: 保存路径
        """
        if results is None:
            results = self.results
        
        
        if len(fnames) < len(results):
            print('LOG: 没有为每个切割图片命名! 将随机生成文件名.')
            fnames += [randName() for i in range(len(results) - len(fnames))]
        
        for fname, im in zip(fnames, results):
            cv2.imwrite('{}/{}.jpg'.format(path, fname), im)
            
    
    def thresholdRange(self, color_range):
        """
        按颜色二值化
        :param color_range: (np.array, np.array) 表示颜色区间的元组
        """
        self.threshold_im = cv2.inRange(self.raw_im, *color_range)
    
    def thresholdGrey(self, thres):
        """
        按灰度二值化, raw_im中灰度大于阈值的点都转为白色, 不够白的都变黑
        :param thres: 阈值
        """
        grey_im = cv2.cvtColor(self.raw_im, cv2.COLOR_BGR2GRAY)
        _ret, self.threshold_im = cv2.threshold(grey_im, thres, 255, cv2.THRESH_BINARY)
    
    def findContours(self):
        _im, self.contours, _hier = cv2.findContours(
            self.threshold_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    def findRecContours(self):
        self.findContours()
        self.rec_contours = list(map(cv2.boundingRect, self.contours))
        
    def filterRecContoursByArea(self, k=1):
        """
        筛选出面积前k大的矩形轮廓
        self.rec_contours必须是cv2.boundingRect返回的四元元组
        """
        self.rec_contours.sort(key=lambda it: it[2] * it[3], reverse=True)  # 根据面积排序
        self.rec_contours = self.rec_contours[:k]
        
    def changePerspective(self, arr1, arr2, size):
        """
        :param arr1: src 四点坐标
        :param arr2: dst 四点坐标
        :param size: 变换后图像的尺寸
        """
        H = cv2.getPerspectiveTransform(np.array(arr1, dtype=np.float32),
                                         np.array(arr2, dtype=np.float32))
    
        self.aligned_im = cv2.warpPerspective(self.threshold_im, H, size)
    
    def cropRecContours(self, recs, base_im=None, pad=0):
        if base_im is None:
            base_im = self.raw_im
        
        self.results = []
        for x, y, w, h in recs:
            x0, y0, x1, y1 = map(int, (x + pad * w, y + pad * h, x + w - pad * w, y + h - pad * h))
            # print((x0, y0), (x1, y1))
            self.results.append(base_im[y0:y1, x0:x1])
    
    def work(self, im):
        self.getRawIm(im)


class LightCropper(Cropper):
    def work(self, im):
        super(LightCropper, self).work(im)
        self.thresholdRange(RED)
        self.findRecContours()
        self.filterRecContoursByArea(5)
        self.rec_contours.sort()
        self.cropRecContours(self.rec_contours, self.threshold_im)
        
        if SAVE:
            self.saveCropResults()
        
        return self.results
        
        
class HandCropper(Cropper):
    def filterSudoku(self):
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
                    w, h = h, w  # PP 一开始试图修改rec造成遗忘，   规定：尽量不要用下标
                    break
        
            # return abs(w / h - ASPECT_RATIO) < ASPECT_RATIO * PROPORTION
            return (any(abs(abs(angle) - d) < ROTATE_BOUND for d in directions) and
                    abs(w / h - ASPECT_RATIO) < ASPECT_RATIO * PROPORTION and
                    SUDOKU_WIDTH * LOW_THRESHOLD < w < SUDOKU_WIDTH * HIGH_THRESHOLD and
                    SUDOKU_HEIGHT * LOW_THRESHOLD < h < SUDOKU_HEIGHT * HIGH_THRESHOLD)
    
        self.minAreaRects = list(map(cv2.minAreaRect, self.contours))  # array([(cx, cy), (w, h), angle])
        self.minAreaRects = list(filter(legal, self.minAreaRects))  # 过滤过度旋转的矩形
        self.minAreaRects.sort(key=lambda it: it[1][0] * it[1][1], reverse=True)
        self.minAreaRects = self.minAreaRects[:15]  # 选出面积前15大的矩形

        if len(self.minAreaRects) < 9:
            pass  # todo 异常处理

        # 聚集筛选
        dist_sums = []
        for i in range(len(self.minAreaRects)):
            dist_sums.append(sum(dist(self.minAreaRects[i][0], self.minAreaRects[j][0])
                                 for j in range(len(self.minAreaRects)) if i != j))
        self.minAreaRects = [tp[0] for tp in sorted(zip(self.minAreaRects, dist_sums),
                                                    key=lambda it: it[1])[:9]]
    
    @staticmethod
    def resumeOrder(arr, ykey, xkey):
        # 恢复现实顺序
        # 0 1 2
        # 3 4 5
        # 6 7 8
        arr.sort(key=ykey)  # 先按y轴排序
        parts = [sorted(arr[i:i + 3], key=xkey)
                for i in range(0, 9, 3)]  # 每三个一组按x轴排序
        return parts[0] + parts[1] + parts[2]
    
    def alignPlank(self):
        RectCorner = namedtuple('RectCorner', ['lu', 'ru', 'ld', 'rd'])
        raw_corners = []
        for i, rec in enumerate(self.minAreaRects):
            box = [tuple(map(int, b)) for b in cv2.boxPoints(rec)]
            box.sort(key=lambda it: it[1])
            box = sorted(box[:2]) + sorted(box[2:])
            # print('test box orders', box)
            raw_corners.append(RectCorner(*box))  # raw_corners: 九个格子的四个角
        raw_corners = self.resumeOrder(raw_corners,
                                       ykey=lambda it: it.lu[1], xkey=lambda it: it.lu)
    
        sudoku_corners = RectCorner(raw_corners[0].lu, raw_corners[2].ru,
                            raw_corners[6].ld, raw_corners[8].rd)
        sudoku_width = int(max(dist(sudoku_corners.lu, sudoku_corners.ru), dist(sudoku_corners.ld, sudoku_corners.rd)))
        sudoku_height = int(max(dist(sudoku_corners.lu, sudoku_corners.ld), dist(sudoku_corners.ru, sudoku_corners.rd)))
        
        dst = ((0, 0.5 * sudoku_height), (sudoku_width, 0.5 * sudoku_height),
                (0, 1.5 * sudoku_height), (sudoku_width, 1.5 * sudoku_height))
        
        self.changePerspective(sudoku_corners, dst, (sudoku_width, int(1.5 * sudoku_height)))
        
        if DEBUG:
            self.imshow(self.aligned_im, 'align')
        
        # self._debug(self.light)
        
    def splitSudokuLight(self):
        h, w = self.aligned_im.shape
        self.threshold_light_im = self.aligned_im[0: int(1 / 3 * h), :]
        self.threshold_sudoku_im = self.aligned_im[int(1 / 3 * h):, :]
        
        if DEBUG:
            self.imshow(self.threshold_light_im, 'light')
            self.imshow(self.threshold_sudoku_im, 'sudoku')

    def work(self, im):
        super(HandCropper, self).work(im)
        self.thresholdGrey(150)
        self.findContours()
        self.filterSudoku()  # minAreaRect旋转矩形, 筛选出九个九宫格块的轮廓
        self.alignPlank()  # 透视变换, 对齐并切割出大板（包含整个九宫格和七段管数字板）
        self.splitSudokuLight()
        
        self.threshold_im = self.threshold_sudoku_im
        self.findRecContours()
        self.filterRecContoursByArea(9)  # boundingRect, 筛选出九宫格块
        self.rec_contours = self.resumeOrder(self.rec_contours[:9],
                                             ykey=lambda it: it[1], xkey=lambda it: it[0])
        self.cropRecContours(self.rec_contours, base_im=self.threshold_sudoku_im)
        
        if SAVE:
            self.saveCropResults()
            self.saveCropResults(results=[self.threshold_light_im])
            
        return self.results
        

if __name__ == '__main__':
    # d = Detector(LightCropper(), None)
    # d.work('../test_im/52648.jpg')
    
    d = Detector(HandCropper(), None)
    d.work('../test_im/real1.jpg')