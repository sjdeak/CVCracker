import os, cv2, operator
import numpy as np
from args import TRAIN_SIZE


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


if __name__ == '__main__':
    # from_cv_digit()
    from_train_hand()