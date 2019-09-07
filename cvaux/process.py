"""图像处理"""


def resize_and_thresh(im, size, fx=0, fy=0):
    """
    只是包装opencv函数，没有改动接口
    对黑白二值图像进行缩放和二值化，从而保证结果仍是黑白二值图
    :param size: =(0, 0)时按照fx和fy推算
    size和fx, fy至少有一方不为零
    """
    im = cv2.resize(im, size, fx=fx, fy=fy)
    ret, im = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)
    return im
