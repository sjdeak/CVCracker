import cv2

class Recognizer:
    def __init__(self, imname):
        self.raw_im = cv2.imread(imname)
        self.result = []

        self.raw_im_process()
        self.find_recs()
        self.filter_and_resort_recs()
        self.loop_process(func=self.single_recognize)

    def raw_im_process(self):
        pass

    def find_recs(self):
        im, self.contours, hier = cv2.findContours(
            self.im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.recs = list(map(cv2.boundingRect, self.contours))

    def filter_and_resort_recs(self):
        pass

    def loop_process(self, func):
        for x, y, w, h in self.recs:
            (x0, y0), (x1, y1) = (x, y), (x + w, y + h)
            self.result.append(func(self.im[y0:y1, x0:x1]))

    def single_recognize(self, im):  # 此处im已经被二值化处理
        pass