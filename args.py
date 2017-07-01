import numpy as np

# 视频颠倒时用read_invert_frame

# hand.py
ROTATE_BOUND = 10  # 最多旋转几度
ASPECT_RATIO = 28 / 16  # 标准宽高比
PROPORTION = 1 / 5  # 和标准宽高比的最大差别界限

LOW_THRESHOLD = 0.6
HIGH_THRESHOLD = 1.4

SODOKU_WEIGHT = 50  # 非常重要的两个参数，需要实际测量填写
SODOKU_HEIGHT = 28  # 当前宽高估计值

MATERIAL_FILE = 'trained_knn_models/full_train_hand.npz'  # 整理好了的训练样本文件地址
TRAIN_SIZE = (100, 55)

# test.py
VIDEO = 'test_im/real_video_part.m4v'

# light.py
RED = (np.array([0, 0, 210]), np.array([255, 255, 255]))

# train.py
# TRAIN_SIZE