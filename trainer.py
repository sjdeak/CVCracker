import os, cv2, shutil
from hand import HandRecognizer
from othertools import rand_name
from args import TRAIN_SIZE

class HandTrainer(HandRecognizer):
    def __init__(self, imname):
        self.acNums = os.path.basename(imname).split('.')[0]
        HandRecognizer.__init__(self, imname)

    def loop_process(self, func, pad=0):
        pad = 0.05
        # self._debug(self.im)
        for i, (x, y, w, h) in enumerate(self.recs):
            x0, y0, x1, y1 = map(int, (x + pad * w, y + pad * h, x + w - pad * w, y + h - pad * h))
            # print(i, x0, y0, x1, y1, self.im.shape)
            func(self.acNums[i], self.im[y0:y1, x0:x1])


    def single_recognize(self, acNum, im):
        im = cv2.resize(im, TRAIN_SIZE)
        cv2.imwrite('train_hand/{}/{}.jpg'.format(acNum, rand_name()), im)


class VideoHandTrainer(HandTrainer):
    def __init__(self, im, acNums):
        self.acNums = acNums
        HandRecognizer.__init__(self, im, already_read=True)


if __name__ == '__main__':
    HandTrainer('material_hand/148623975.jpg')