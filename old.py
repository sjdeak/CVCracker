import cv2
import numpy as np
from hand import HandRecognizer
from args import OLD_FONT_THRESHOLD


class OldRecognizer(HandRecognizer):
    def loop_process(self, func):
        pad = 0.05
        for x, y, w, h in self.recs:
            x0, y0, x1, y1 = map(int, (x + pad * w, y + pad * h, x + w - pad * w, y + h - pad * h))
            single = cv2.cvtColor(self.raw_sudoku_im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            self.result.append(func(single))

    def single_recognize(self, im):
        ret, im = cv2.threshold(im, OLD_FONT_THRESHOLD, 255, cv2.THRESH_BINARY)

        # self._debug(im)

        r, c = im.shape
        edges = cv2.Canny(im, 20, 50)

        black, mid_area = 0, 0
        for i in range(r):
            for j in range(c):
                if edges[i, j] == 255:
                    black += 1
                    if 1 / 3 * c < j < 2 / 3 * c:
                        mid_area += 1

        return mid_area / black  # 中间黑色像素占整个中间部分的比例


    def get_final_result(self):
        # print(*zip(self.result, range(9)), sep='\n')
        return sorted(zip(self.result, range(9)), reverse=True)[0][1]


if __name__ == '__main__':
    print(OldRecognizer('test_im/raw_old.jpg').get_final_result())