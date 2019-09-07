"""基础函数"""


def get_color_type(im):
    if im.shape == 2:
        return 'GREY'
    else:
        return 'BGR'

def show(self, im, title='dummy'):
    """显示im"""
    cv2.imshow(title, im)
    cv2.waitKey(0)