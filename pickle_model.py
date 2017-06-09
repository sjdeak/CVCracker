import cv2, pickle
import numpy as np

def from_cv_digit():
    train_im = cv2.imread('raw_train_materials/digits.png', 0)
    x = np.array([np.hsplit(row, 100) for row in np.vsplit(train_im, 50)])  # 切割开素材
    train_data = x[5:, :].reshape(-1, 400).astype(np.float32)  # (4500, 400)

    k = np.arange(1, 10)
    train_label = np.repeat(k, 500)[:, np.newaxis]  # (4500, 1)

    np.savez('cv_dight.npz', train_data=train_data, train_label=train_label)


def from_train_hand():
    pass

if __name__ == '__main__':
    from_cv_digit()