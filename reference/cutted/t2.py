import os, sys, cv2

cutted = cv2.imread('cutted3_raw_size.jpg', 0)
#off = cv2.imread('cutted3.jpg', 0)

off = cv2.imread(os.path.expanduser('~/Desktop/off/3/_00486.png_f468e33bb40b935e92fc27a6b47a096e_raw.jpg'))

for i in range(55):
    for j in range(100):
        if cutted[i, j] == 0:
            off[i, j] = 100

cv2.imshow('hi', off)
cv2.waitKey(0)