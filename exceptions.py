class LightImError(Exception): pass


class LightRecFail(Exception):
    def __init__(self, id, info):
        """
        :param id: 该数字出错前有几个数字是成功识别的
        :param info: 出错信息
        """
        self.id = id
        self.info = info