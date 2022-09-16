# Linux端基础训练推理功能测试

Linux端基础训练推理功能测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 | 多机多卡 |
| -------- | -------- | -------- | -------- | -------- |
| NAFNet   | NAFNet   | 正常训练 | -        | -        |

- 推理相关：

| 算法名称 | 模型名称 | 模型类型 | device  | batchsize |
| -------- | -------- | -------- | ------- | --------- |
| NAFNet   | NAFNet   | 正常模型 | GPU/CPU | 1         |

## 2 目录介绍

```
test_tipc
    |--configs                              # 配置目录
        |--NAFNet                           # 模型名称
            |--train_infer_python.txt       # 基础训练推理测试配置文件
    |--docs
        |--train_infer_python.md            # TIPC说明文档
    |--data                                 # 推理数据
    |--output                               # TIPC推理结果与日志
    |--test_train_inference_python.sh       # TIPC基础训练推理测试解析脚本
    |--common_func.sh                       # TIPC基础训练推理测试常用函数
    |--prepare.sh                           # 推理数据准备脚本
```

## 3 测试流程

### 3.1 准备数据

用于基础训练推理测试的数据位于`test_tipc/data/`，可在项目根目录下运行以下命令解压：

```shell
bash test_tipc/prepare.sh ./test_tipc/configs/NAFNet/train_infer_python.txt 'lite_train_lite_infer'
```

### 3.2 准备环境

- 安装 PaddlePaddle == 2.2.2

- 安装 AutoLog（规范化日志输出工具）

  ```
  pip install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
  ```

### 3.3 功能测试

使用本工具，可以测试不同功能的支持情况，以及预测结果是否对齐，测试流程概括如下：

1. 运行`prepare.sh` 准备测试所需数据和模型；
2. 在项目根目录下，运行要测试的功能对应的测试脚本 `test_train_inference_python.sh`

```shell
# 功能：准备数据
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/prepare.sh ./test_tipc/configs/NAFNet/train_infer_python.txt 'lite_train_lite_infer'

# 功能：运行测试
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/NAFNet/train_infer_python.txt 'lite_train_lite_infer'
```
