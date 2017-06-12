# 运行环境
- OpenCV 3.2
- Python 3.6.1
- numpy 1.12.1
- pyserial 3.3

# 使用方法
## 1-预先配置部分参数
- `SODOKU_WIDTH`
- `SODOKU_HEIGHT`

## 2-准备训练素材
用摄像头录一段大神符录像后运行`crop_material.py`

## 3-实际使用
1. 把`VIDEO`参数设置为0
2. 上位机中运行`main.py`
3. 操作手观察到九宫格中的手写数字出现时按`Enter`键
4. 如果没有战车没有反应则等九宫格刷新完毕后重复第2步

# 程序内部结构

# 下位机通信协议
