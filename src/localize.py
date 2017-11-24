import cv2
import numpy as np
from hand import HandRecognizer


class Localizer(HandRecognizer):
    def __init__(self, imname, already_read=False):
        self.raw_im = imname if already_read else cv2.imread(imname)
        self.im = None
        self.recs = None

        self.raw_im_process()
        self.find_contours()  # 找出所有轮廓
        self.filter_contours()

        self.cal_value()
        self.move_value = self.get_move_value()

    def cal_value(self):
        self.recs = self.resume_order(self.recs, ykey=lambda it: it[0][1], xkey=lambda it: it[0])

        vch, vcw = map(int, map(lambda it: it // 2, self.raw_im.shape[:2]))  # vison center height/width
        bcw, bch = map(int, self.recs[4][0])  # board center width/height

        self.vision_center = (vch, vcw)
        self.board_center = (bch, bcw)


    def get_move_value(self):
        """
        :return: (x偏移量, y偏移量)
        x: >0表示右移 <0左移
        y: >0表示上移 <0下移
        """

        vch, vcw = self.vision_center
        bch, bcw = self.board_center

        return bcw - vcw, vch - bch

if __name__ == '__main__':
    print(Localizer('test_im/wrong1.jpg').move_value)