import numpy as np
import cv2

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


def check_line(pos, direction, im):
    h, w = im.shape
    dis = im.shape[1]//6  # 宽度的1/4

    for i in range(-dis, dis):
        p = list(pos)
        p[direction] += i

        nr, nc = p
        if not (nr in range(h) and nc in range(w)):
            continue
        if im[nr, nc] == 255:
            return 1
    else:
        return 0

def solve(im):
    h, w = im.shape

    if h / w > 2.5:
        return 1

    ch, cw = (h // 2, w // 2)
    offset = w // 10

    a, d = (0, cw), (h, cw)
    b, f = (ch//2, offset), (ch//2, w)
    c, e = (ch + ch//2, 0), (ch + ch//2, w-offset)
    g = (ch, cw)

    VER, HOR = 0, 1
    checkpoints = [a, b, c, d, e, f, g]
    check_directions = [VER, HOR, HOR, VER, HOR, HOR, VER]

    # print(checkpoints)

    result = tuple(check_line(*it, im)
                   for it in zip(checkpoints, check_directions))

    try:
        return DIGIT[result]
    except KeyError:
        print(result)
        return 'Error!'

if __name__ == '__main__':
    raw_im = cv2.imread('test_im/bad_light1.jpg')


    im = cv2.inRange(raw_im, np.array([0, 0, 210]), np.array([255, 255, 255]))

    # cv2.imshow('二值化', im)

    im, contours, hier = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    recs = list(map(cv2.boundingRect, contours))
    # 根据轮廓矩形面积排序
    contours = sorted(zip(contours, recs), key=lambda it: it[1][2] * it[1][3], reverse=True)


    light_pos = []
    for cnt in contours[:5]:
        x, y, w, h = cnt[1]
        light_pos.append(((x, y), (x + w, y + h)))
    light_pos.sort()

    for it in light_pos:
        (x0, y0), (x1, y1) = it
        print(solve(im[y0:y1, x0:x1]))
