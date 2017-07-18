"""切割从视频中切割并分类九宫格图片，为之后的训练准备素材"""
import os, cv2, shutil
import numpy as np
from trainer import VideoHandTrainer
from pyaux.others import rand_name
from args import TRAIN_SIZE


# MATERIAL_VIDEO = 'raw_train_materials/material.m4v'
MATERIAL_VIDEO = 'test_im/real_video.m4v'
INIT_MODE = False  # INIT_MODE下会先清空以前切割的文件


def clear_dir():
    shutil.rmtree('train_hand')
    os.mkdir('train_hand')
    for i in range(1, 10):
        os.mkdir('train_hand/{}'.format(i))


def crop_materials():
    if INIT_MODE:
        clear_dir()

    cap = cv2.VideoCapture(MATERIAL_VIDEO)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = frame[::-1, ::-1]

        cv2.imshow('frame', frame)
        response = cv2.waitKey(100)
        if response == ord('q'):
            break
        if response == ord('t'):
            print('请输入九宫格块的实际数字')
            # data = '123456789'
            data = input()
            if data != 'pass':
                acNums = list(map(int, data))
                VideoHandTrainer(frame, acNums)


def from_cv_digit():
    train_im = cv2.imread('raw_train_materials/digits.png', 0)
    x = np.array([np.hsplit(row, 100) for row in np.vsplit(train_im, 50)])  # 切割开素材
    train_data = x[5:, :].reshape(-1, 400).astype(np.float32)  # (4500, 400)

    k = np.arange(1, 10)
    train_label = np.repeat(k, 500)[:, np.newaxis]  # (4500, 1)

    np.savez('trained_knn_models/cv_dight.npz', train_data=train_data, train_label=train_label)


def from_train_hand():
    train_data, train_label = [], []

    for dirpath, dirnames, filenames in os.walk('train_hand'):
        if dirpath == 'train_hand':
            continue
        acNum = int(dirpath[-1])
        for fp in filenames:
            if fp.endswith('.jpg'):
                print(os.path.join(dirpath, fp))
                im = cv2.imread(os.path.join(dirpath, fp), 0)
                train_data.append(im.reshape(-1, im.size))
                train_label.append(acNum)

    train_data = np.vstack(train_data).astype(np.float32)
    train_label = np.array(train_label).reshape(-1, 1)

    np.savez('trained_knn_models/train_hand.npz', train_data=train_data, train_label=train_label)

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

def from_official():
    train_data, train_label = [], []

    for dirpath, dirnames, filenames in os.walk('number'):
        if dirpath == 'number':
            continue
        acNum = int(dirpath[-2])
        for fp in filenames:
            if fp.endswith('.png'):
                print(os.path.join(dirpath, fp))
                im = cv2.imread(os.path.join(dirpath, fp), 0)
                im = resize_and_thresh(im, TRAIN_SIZE)
                cv2.imwrite(os.path.expanduser('~/Desktop/off/{}/{}.jpg'.format(acNum, rand_name())), im)

                train_data.append(im.reshape(-1, im.size))
                train_label.append(acNum)

    train_data = np.vstack(train_data).astype(np.float32)
    train_label = np.array(train_label).reshape(-1, 1)

    np.savez('trained_knn_models/official.npz', train_data=train_data, train_label=train_label)


if __name__ == '__main__':
    # crop_materials()
    from_official()
    # from_cv_digit()
    # from_train_hand()