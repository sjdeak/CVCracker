import math
import numpy as np
import hashlib, time


def dist(p1, p2):
    """
    计算任意维度中两点间距离
    :param p1, p2: 等长序列
    :return: 距离
    """
    vec = np.array(p1) - np.array(p2)
    return math.sqrt(vec.dot(vec))

def randName():
    """随机生成文件名"""
    return hashlib.md5(str(time.time()).encode()).hexdigest()