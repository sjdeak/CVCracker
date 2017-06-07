import pickle

import matplotlib.pyplot as plt
import numpy as np
from learnknn import my_knn_class

from archives.learnknn import pca_imtools

# 训练样本
with open('points_normal_1.pkl', 'rb') as f:
    class_1 = pickle.load(f)
    class_2 = pickle.load(f)
    labels = pickle.load(f)

model = my_knn_class.KNNClassifier(labels, np.vstack((class_1, class_2)))

# 测试数据
with open('points_normal_2.pkl', 'rb') as f:
    class_1 = pickle.load(f)
    class_2 = pickle.load(f)
    labels = pickle.load(f)

# print(model.classify(class_1[0]))

def classify(x, y, model=model):
    """
    批量分类函数，返回所有点的分类结果组成的数组
    """
    return np.array([model.classify([xx, yy]) for (xx, yy) in zip(x, y)])


pca_imtools.plot_2D_boundary([-6, 6, -6, 6], [class_1, class_2], classify, [1, -1])
plt.show()