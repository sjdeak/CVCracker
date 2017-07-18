import hashlib, time


def rand_name():
    """
    随机生成文件名
    """
    return hashlib.md5(str(time.time()).encode()).hexdigest()