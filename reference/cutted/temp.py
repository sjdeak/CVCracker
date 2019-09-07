import os, sys, cv2

cutted = cv2.imread('cutted3x.jpg', 0)

res = []
for dirpath, dnames, fnames in os.walk(os.path.expanduser('~/Desktop/off')):
    for fname in fnames:
        if not fname.endswith('.jpg'):
            continue
        fullname = os.path.join(dirpath, fname)
        cmp_im = cv2.imread(fullname, 0)
        dif = [1 for i in range(55) for j in range(100) if cutted[i, j] == cmp_im[i, j] and cutted[i, j] == 0]
        res.append((len(dif), fullname))
        
print(*sorted(res, reverse=True)[:20], sep='\n')