import cv2, pickle
import numpy as np
from numpy.random import randn
from math import sqrt

import matplotlib.pyplot as plt


def cal_dist(p1, p2):
    return sqrt(sum((p1 - p2) ** 2))


class KNNClassifier(object):
    def __init__(self, labels, samples):
        self.labels = labels  # 标记样本的实际分类
        self.samples = samples  # 样本

    def classify(self, point, k=3):  # 分类point点
        dist = np.array([cal_dist(point, s) for s in self.samples])

        ndx = dist.argsort()

        votes = {}
        for i in range(k):
            label = self.labels[ndx[i]]
            votes.setdefault(label, 0)
            votes[label] += 1

        return max(votes.keys(), key=lambda it: votes[it])
