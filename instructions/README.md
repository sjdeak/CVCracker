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


# TODO List

- hand手工训练
- light优化
- weird num 冲突处理
- 串口

- 半自动化：操作手处理后两位
- 各数字黑色像素点数统计，用于猜测数字
- 自定义异常
- 文档、注释

- localizer验证
- 换上官方素材
- light.py HSV过滤红色
- 参数管理模块
- 小符模式
- main.py 主循环
- linux下跑一遍

