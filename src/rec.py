import functools
import cv2


class LightRecognizer:
    DIGIT = {
        (1, 0, 1, 1, 0, 1, 1): 2,
        (1, 0, 0, 1, 1, 1, 1): 3,
        (0, 1, 0, 0, 1, 1, 1): 4,
        (1, 1, 0, 1, 1, 0, 1): 5,
        (1, 1, 1, 1, 1, 0, 1): 6,
        (1, 0, 0, 0, 1, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 0, 1, 1, 1, 1): 9
    }
    
    @staticmethod
    def checkLine(pos, direction, im):
        h, w = im.shape
        dis = w // 6  # 宽度的1/4
        cnt = 0

        for i in range(-dis, dis):
            p = list(pos)
            p[direction] += i

            nr, nc = p
            if not (nr in range(h) and nc in range(w)):
                continue

            if im[nr, nc] == 255:
                im[nr, nc] = 100
                cnt += 1
            else:
                im[nr, nc] = 100

            if cnt >= 2:
                return 1
        else:
            return 0
    
    @multiple
    def singleRecognize(self, im):
        im = cv2.resize(im, None, fx=3, fy=3)
        ret, im = cv2.threshold(im, 150, 255, cv2.THRESH_BINARY)
        h, w = im.shape

        if h / w > 2.5:
            return 1

        ch, cw = (h // 2, w // 2)
        offset = w // 8

        a, d = (0, cw), (h, cw - offset)
        b, f = (ch // 2, offset), (ch // 2, w)
        c, e = (ch + ch // 2, 0), (ch + ch // 2, w - offset)
        g = (ch, cw -  offset // 2)

        ver, hor = 0, 1
        checkpoints = [a, b, c, d, e, f, g]
        check_directions = [ver, hor, hor, ver, hor, hor, ver]

        res = tuple(self.checkLine(*it, im)
                       for it in zip(checkpoints, check_directions))
        # self._debug(im)

        try:
            return LightRecognizer.DIGIT[res]
        except KeyError:
            print('ERROR: 七段管识别出错')
            
    def multipleRecognize(self, ims):  # # 绑定方法在被装饰后变成了普通函数
        return multiple(self.singleRecognize)(ims)  # 解决类方法的装饰问题
    

def multiple(fin):
    @functools.wraps(fin)
    def wrapper(images):
        return [fin(image) for image in images]
    return wrapper


if __name__ == '__main__':
    pass
    # lc = LightCropper()
    # lc.work('../test_im/52648.jpg')
    # lr = LightRecognizer()
    # print(lr.multipleRecognize(lc.result))