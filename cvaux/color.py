"""
opencv辅助函数库
"""

BGR_RED = (0, 0, 255)
HSV_RED_BOUND1 = (np.array([0, 43, 46]), np.array([10, 255, 255]))
HSV_RED_BOUND2 = (np.array([156, 43, 46]), np.array([180, 255, 255]))

def filter_red(raw_im):	
	"""基于HSV的色彩过滤"""
	hsv = cv2.cvtColor(raw_im, cv2.COLOR_BGR2HSV)
	red1 = cv2.inRange(hsv, *HSV_RED1_BOUND1)
	red2 = cv2.inRange(hsv, *HSV_RED2_BOUND2)

	r, c, dummy = raw_im.shape
	im = np.array([255 if red1[i,j] == 255 or red2[i,j] == 255 else 0
	 for i in range(r) for j in range(c)], dtype='uint8').reshape((r, c))

	return im