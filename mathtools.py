import numpy as np
import math


def dist(p1, p2):
    """
    计算两点间距离
    :param p1, p2: 等长序列
    :return: 距离
    """
    vec = np.array(p1) - np.array(p2)
    return math.sqrt(vec.dot(vec))