import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('digits.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

x = np.array(cells)  # 50x100张图片，每个数字各500张 (50,100,400)

train_data = x[:, :50].reshape(-1, 400).astype(np.float32)  # (2500, 400)
test_data = x[:, 50:].reshape(-1, 400).astype(np.float32)

k = np.arange(10)
train_label = np.repeat(k, 250)[:, np.newaxis]  # (2500, 1)
test_label = train_label.copy()

knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_label)
retval, results, neign_resp, dists = knn.findNearest(test_data, 3)

matches = results == test_label
correct = np.count_nonzero(matches)

print(correct / matches.size * 100, '%', sep='')