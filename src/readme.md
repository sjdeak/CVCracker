# 接口
Light/Hand Cropper 接收已经打开的图片  返回切割好的二值化图片列表
Light/Hand Recognizer multipleRecognize接收切割好的二值化图片列表  返回识别结果（数字）列表

def multiple(fin):
    @functools.wraps(fin)
    def wrapper(images):
        return [fin(image) for image in images]
    return wrapper