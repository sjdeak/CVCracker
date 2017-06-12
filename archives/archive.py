trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)
responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)

red = trainData[responses.ravel() == 0]
plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')

blue = trainData[responses.ravel() == 1]
plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')


newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)
plt.scatter(newcomer[:, 0], newcomer[:, 1], 80, 'g', 'o')

knn = cv2.ml.KNearest_create()
knn.train(trainData, responses)

ret, results, neighbours, dist = knn.find_nearest(newcomer, 3)

print(results)
print(neighbours)
print(dist)

plt.show()


import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('digits.png', 0)

cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

x = np.array(cells)  # 50x100张图片，每个数字各500张 (50,100,400)

train_data = x[5:, :].reshape(-1, 400).astype(np.float32)  # (4500, 400)

k = np.arange(1, 10)
train_label = np.repeat(k, 500)[:, np.newaxis]  # (4500, 1)

self.knn = cv2.ml.KNearest_create()
self.knn.train(train_data, cv2.ml.ROW_SAMPLE, train_label)


