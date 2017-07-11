import numpy as np
from configparser import ConfigParser

# hand.py
ROTATE_BOUND = 10  # 最多旋转几度
ASPECT_RATIO = 28 / 16  # 标准宽高比
PROPORTION = 1 / 5  # 和标准宽高比的最大差别界限

LOW_THRESHOLD = 0.6
HIGH_THRESHOLD = 1.4

HAND_FONT_THRESHOLD = 130
OLD_FONT_THRESHOLD = 170

cfg = ConfigParser()
cfg.read('config.ini', encoding='utf8')

SUDOKU_WIDTH = cfg.getint('sudoku', 'sudoku_width', fallback=50)

SUDOKU_HEIGHT = cfg.getint('sudoku', 'sudoku_height', fallback=28)

MATERIAL_FILE = 'trained_knn_models/full_train_hand.npz'  # 整理好了的训练样本文件地址
TRAIN_SIZE = (100, 55)

# test.py
# VIDEO = 'test_im/real_video_part.m4v'
VIDEO = 'raw_train_materials/material.m4v'

# light.py
RED = (np.array([0, 0, 210]), np.array([255, 255, 255]))