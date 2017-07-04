# 运行环境
- OpenCV 3.2
- Python 3.6.1
- numpy 1.12.1
- pyserial 3.3

# 使用方法
## 1-预先配置部分参数
- `SODOKU_WIDTH`
- `SODOKU_HEIGHT`
- `TRAIN_SIZE`

## 2-准备训练素材
用摄像头录一段大神符录像后运行`crop_material.py`

## 3-实际使用
1. 把`VIDEO`参数设置为0
2. 上位机中运行`main.py`
3. 操作手观察到九宫格中的手写数字出现时按`Enter`键
4. 如果没有战车没有反应则等九宫格刷新完毕后重复第2步

# 程序内部结构

# 下位机通信协议

# ideas
- 半自动化：操作手处理后两位
- 各数字黑色像素点数统计，用于猜测数字

# TODO List

- hand手工训练 √
- light优化 √
- weird num 冲突处理 √

- localizer验证  √
- 换上官方素材  √ 效果不好
- light.py HSV过滤红色  √ 效果不好
- 参数管理模块  √
- 为什么官方素材效果不好？ 水平的旋转可以通过仿射变换解决，俯仰视
    - 解决办法： **现场实地训练**
- 用库组织辅助函数
- 小符模式
- main.py 主循环
- linux下跑一遍

- 多线程....
- 串口
- 自定义异常
- 文档、注释

# pits
1. 无法从版本库中删除workspace.xml

git rm 把工作树上的workspace.xml删除了

workspace.xml一修改，pycharm自动add

然后再commit....完了