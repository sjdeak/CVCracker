import os, sys, cv2
import numpy as np
from crop import *
from rec import *
from args import RED


class Detector:
    def __init__(self, cropper, recognizer):
        self.cropper = cropper
        self.recognizer = recognizer
        
        self.raw_im = None
        self.crop_result = None
        self.recognize_resule = None
    
    def performCrop(self):
        self.crop_result = self.cropper.work(self.raw_im)
    
    def perfromRecognize(self):
        print(self.recognizer.multipleRecognize(self.crop_result))
        
    def imread(self, im):
        """
        读入图片
        :param im: 图片文件路径 或 已经读入的图片, 可自动识别
        """
        if os.path.isfile(im):
            self.raw_im = cv2.imread(im)
            print('LOG: 已从路径{}读入图片'.format(im))
        elif type(im) == np.ndarray:
            self.raw_im = im
        else:
            print('Wrong im')
            return
        
    def work(self, im):
        self.imread(im)
        self.performCrop()
        self.perfromRecognize()
        
            
if __name__ == '__main__':
    d = Detector(LightCropper(), LightRecognizer())
    d.work('../test_im/52648.jpg')