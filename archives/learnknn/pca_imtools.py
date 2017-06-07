import os
import numpy as np
import matplotlib.pyplot as plt

def get_imlist(path):
    """
    返回目录中所有jpg文件的地址列表
    :param path: 目录名 
    :return: 图片地址列表
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


def plot_2D_boundary(plot_range, points, decisionfcn, labels, values=[0]):
    """
    :param plot_range: (xmin, xmax, ymin, ymax) 
    :param points: 点集组成的序列
    :param decisionfcn: 分类函数
    :param labels: 实际正确结果
    :param values: 
    """
    # x = np.arange(plot_range[0], plot_range[1], .1)
    # y = np.arange(plot_range[2], plot_range[3], .1)
    # xx, yy = np.meshgrid(x, y)
    # xxx, yyy = xx.flatten(), yy.flatten()
    #
    # zz = np.array(decisionfcn(xxx, yyy)).reshape(xx.shape)
    #
    # plt.contour(xx, yy, zz, values)

    clist = ['b', 'g', 'r', 'k', 'm', 'y']

    # 实际分类结果
    for i in range(len(points)):
        d = decisionfcn(points[i][:, 0], points[i][:, 1])

        correct_ndx = (labels[i] == d)
        incorrect_ndx = (labels[i] != d)

        plt.plot(points[i][correct_ndx, 0], points[i][correct_ndx, 1], '*', color=clist[i])
        plt.plot(points[i][incorrect_ndx, 0], points[i][incorrect_ndx, 1], 'o', color=clist[i])

    plt.axis('equal')
