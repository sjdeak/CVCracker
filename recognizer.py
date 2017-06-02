import cv2

class Recognizer:
    def __init__(self, imname):
        """
        :param imname: 待处理图片的文件路径 
        """
        self.raw_im = cv2.imread(imname)
        self.im = None
        self.recs = None
        self.result = []  # result: 存放最终识别结果

        self.raw_im_process()
        self.find_contours()  # 找出所有轮廓
        self.find_target_recs()
        self.loop_process(func=self.single_recognize)

    def raw_im_process(self):
        """原始图像预处理"""
        pass

    def find_contours(self):
        im, self.contours, hier = cv2.findContours(
            self.im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def find_target_recs(self):
        """
        筛选出目标矩形
        Light_recognizer: 五个数字的轮扣矩形
        Hand_recognizer: 九宫格内的九个矩形格子
        """
        pass

    def loop_process(self, func, pad=0):
        for x, y, w, h in self.recs:
            x0, y0, x1, y1 = map(int, (x + pad * w, y + pad * h, x + w - pad * w, y + h - pad * h))
            # print((x0, y0), (x1, y1))
            self.result.append(func(self.im[y0:y1, x0:x1]))

    def single_recognize(self, im):
        """
        识别单个数字
        :param im: 从二值化后的原图中切割出的单个数字图片  
        """
        pass

    def _debug(self, im):
        """调试函数，显示self.im"""
        cv2.imshow('dummy', im)
        cv2.waitKey(0)