import cv2
import numpy as np
from hand import HandRecognizer


class OldRecognizer(HandRecognizer):
    def single_recognize(self, im):
        # todo HandRecognizer方案尚未确定，不知道最终方案里此时传入的im是否被二值化

        if not hasattr(self.single_recognize, 'counter'):
            self.single_recognize.counter = 0
        else:
            self.single_recognize.counter += 1

        r, c = im.shape
        edges = cv2.Canny(im, 20, 50)

        black, mid_area = 0, 0
        for i in range(r):
            for j in range(c):
                if edges[i, j] == 255:
                    black += 1
                    if 1 / 3 * c < j < 2 / 3 * c:
                        mid_area += 1

        return mid_area / black, self.single_recognize.counter


    def get_final_result(self):
        return sorted(self.result)[0][1]


if __name__ == '__main__':
    print(OldRecognizer('test_im/raw_old.jpg').get_final_result())