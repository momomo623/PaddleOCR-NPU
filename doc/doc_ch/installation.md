## 快速安装

## 推理

```
shapely
numpy >= 1.24
pillow
pyclipper
opencv-python >== 4.1.0
pytorch >= 1.13
```

## 模型转换

```
shapely
numpy >= 1.24
pillow
pyclipper
opencv-python >== 4.1.0
pytorch >= 1.13
paddlepaddle >= 2.0.0
```

**安装PaddlePaddle >= 2.0**

```bash
pip3 install --upgrade pip

# 如果您的机器安装的是CUDA9或CUDA10，请运行以下命令安装
python3 -m pip install paddlepaddle-gpu==2.0.0 -i https://mirror.baidu.com/pypi/simple

# 如果您的机器是CPU，请运行以下命令安装
python3 -m pip install paddlepaddle==2.0.0 -i https://mirror.baidu.com/pypi/simple

# 更多的版本需求，请参照[安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。
```

