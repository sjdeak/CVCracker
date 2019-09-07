"""
在GUI中绘制图形
"""
from basic import *

def putText(im, text, pos):
	"""
	:param text: 要写的文本
	:param pos: 文本左下角坐标
	:return: 写上文本后的新图像
	"""

	COLOR = 100 if get_color_type(im) == 'GREY' else (0, 0, 255)  # 灰度图画黑灰色，彩色图画红色

	ret_im = im.copy()
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(ret_im, text, pos, font, 1, COLOR, 2)
	return ret_im


def debug_rects_center(im, recs):
	"""
	画出覆盖矩形的中心，并显示
	用于调试minAreaRect
	"""
	COLOR = 100 if get_color_type(im) == 'GREY' else (0, 0, 255)
		
	tmp = im.copy()
	for (cx, cy), (w, h), angle in recs:
		tmp = cv2.circle(tmp, (int(cx), int(cy)), 2, COLOR)
	show(tmp)