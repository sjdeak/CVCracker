import pickle
import numpy as np
from numpy.random import randn


n = 200
labels = np.hstack((np.ones(n), -np.ones(n)))

class_1 = 0.6 * randn(n, 2)
class_2 = 1.2 * randn(n, 2) + np.array([5,1])

with open('points_normal_2.pkl', 'wb') as f:
    pickle.dump(class_1, f)
    pickle.dump(class_2, f)
    pickle.dump(labels, f)


class_1 = 0.6 * randn(n, 2)
# 生成环状数据组class_2
r = 0.8 * randn(n, 1) + 5
angle = 2 * np.pi * randn(n, 1)
class_2 = np.hstack((r * np.cos(angle), r * np.sin(angle)))


with open('points_ring_2.pkl', 'wb') as f:
    pickle.dump(class_1, f)
    pickle.dump(class_2, f)
    pickle.dump(labels, f)
