# **NAFNet**
**The paddlepaddle  implemention of NAFNet**

[Simple Baselines for Image Restoration](https://arxiv.org/pdf/2204.04676)  论文复现

[官方源码](https://github.com/megvii-research/NAFNet)

[AI Studio 脚本项目地址]()

## 1. 简介

![NAFNet](https://ai-studio-static-online.cdn.bcebos.com/699b87449c7e495f8655ae5ac8bc0eb77bed4d9cd828451e8939ddbc5732a704)

NAFNet的网络设计和特点如上图所示，采用带跳过连接的UNet作为整体架构，同时修改了Restormer块中的Transformer模块，并取消了激活函数，采取更简单有效的simplegate设计，运用更简单的通道注意力机制

## 2. 复现精度

验收标准：SIDD PSNR: 40.30 SSIM: 0.961 

复现结果：SIDD PSNR: 40.20 SSIM: 0.959

## 3. 数据集、复现模型权重、文件结构

### 数据集

下载数据：

利用[官方代码](https://github.com/megvii-model/HINet#image-restoration-tasks)，得到SIDD数据集并解压到SIDD_Data下


  * ```mkdir ./SIDD_Data ```
  
  * 下载训练集 [train](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php), 将 (./SIDD_Medium_Srgb/Data) 移动到 ./SIDD_Data/
  * 下载测试集 [val](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php)  (ValidationNoisyBlocksSrgb.mat and ValidationGtBlocksSrgb.mat) ,移动到 ./SIDD_Data/
  * 结构如下:
  
    ```bash
    ./SIDD_Data/Data
    ./SIDD_Data/ValidationNoisyBlocksSrgb.mat
    ./SIDD_Data/ValidationGtBlocksSrgb.mat
    ```
  
  * ```python sidd_data_preprocessing.py```
    * 将训练图片剪裁为 512x512 patches.
    * 处理后的数据集可于[AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/149460/0)找到

**注**：为简化操作，可直接使用处理后上传的SIDD数据


### 复现模型权重

AI Studio：[下载链接](https://aistudio.baidu.com/aistudio/datasetdetail/168981)，数据集中的`model_best.pdparams`即为复现得到的最优权重


### 文件结构

```
NAFNet-pd
    |-- dataloaders
    |-- SIDD_Data
         |-- train                 # SIDD-Medium 训练数据
         |-- val                   # SIDD 测试数据
    |-- SIDD_patches
         |-- train_mini            # 小训练数据，用于TIPC测试
         |-- val_mini              # 小测试数据，用于TIPC测试
    |-- logs                       # 训练日志
    |-- test_tipc                  # TIPC: Linux GPU/CPU 基础训练推理测试
    |-- networks
         |-- NAFNet_arch.py        # NAFNet模型代码
    |-- utils                      # 一些工具代码
    |-- config.py                  # 配置文件
    |-- export_model.py            # 预训练模型的导出代码
    |-- infer.py                   # 模型推理代码
    |-- LICENSE                    # LICENSE文件
    |-- losses.py                  # 损失函数
    |-- predict.py                 # 模型预测代码
    |-- README.md                  # README.md文件
    |-- sidd_data_preprocessing.py # SIDD数据预处理代码
    |-- test_denoising_sidd.py     # 测试SIDD数据上的指标
    |-- train.py                   # TIPC训练测试代码
    |-- train_denoising_1card.py   # 单机单卡训练代码
    |-- train_denoising_4cards.py  # 单机多卡训练代码
    |-- training_1card.yml         # 单机单卡训练配置文件
    |-- training_4cards.py         # 单机多卡训练配置文件
```

## 4. 环境依赖

PaddlePaddle == 2.2.2, 若用paddle2.3.2则paddle.cumsum()函数有问题，会带来错误的推理结果

## 5. 快速开始

GPU数量改变时，须保证

`total_batchsize*iter == 8gpus*8bs*400000iters`

与官方保持一致
### 单机单卡

```shell
python train_denoising_1card.py
```

配置文件为 `training_1card.yml`

### 单机四卡

```shell
python -m paddle.distributed.launch train_denoising_4cards.py
```

此处为用四张卡，配置文件为 `training_4cards.yml`


### 日志说明

由于训练模型采用的是脚本任务训练，本身脚本任务就有相应的日志记录，均保存在了`./logs`文件夹下


### 模型评估

在 SIDD 测试数据上作测试

```
python test_denoising_sidd.py --weight ../data/data168981/model_best.pdparams 
```

输出如下：

```
PSNR: 40.2024
SSIM: 0.9590
```


### 模型预测

在 SIDD 小验证集上作预测，结果存放在 `results/` 文件夹下

```
python predict.py --model_ckpt ../data/data168981/model_best.pdparams --data_path ./SIDD_patches/val_mini/ --save_path results/ --save_images
```

输出结果如下：

```
PSNR on test data 40.7486, SSIM on test data 0.9809
```

预测的部分图片如图所示：

| 噪声图像 | 预测输出 |
| --- | --- |
| ![](https://ai-studio-static-online.cdn.bcebos.com/f6379004f10548b28c69e6e841d015594eccdfda204e4581ac40502ac64c2063)| ![](https://ai-studio-static-online.cdn.bcebos.com/597817cf3cc442d3a3d2b733d77924a788f80f0c6e7a4e9cbe38085b00785563)|

### 推理过程：

需要安装 reprod_log：

```
pip install reprod_log
```

模型动转静导出：

```
python export_model.py --model-dir ../data/data168981/model_best.pdparams --save-inference-dir ./inference_output
```

最终在`./inference_output/`文件夹下会生成下面的3个文件：

```
output
  |----model.pdiparams     : 模型参数文件
  |----model.pdmodel       : 模型结构文件
  |----model.pdiparams.info: 模型参数信息文件
```

模型推理：

```
python infer.py --model-dir inference_output --use-gpu True --benchmark False --clean-dir=./SIDD_patches/val_mini/gt_crops/ValidationBlocksSrgb_0.png --noisy-dir=./SIDD_patches/val_mini/input_crops/ValidationBlocksSrgb_0.png
```

输出结果如下：

```
image_name: ./SIDD_patches/val_mini/input_crops/ValidationBlocksSrgb_0.png, psnr: 41.94810134125529
```

## 6. TIPC

首先安装AutoLog（规范化日志输出工具）

```
pip install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
```

在linux下，进入 hinet_paddle 文件夹，运行命令：

```shell
bash test_tipc/prepare.sh ./test_tipc/configs/NAFNet/train_infer_python.txt 'lite_train_lite_infer'
```

```sehll
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/HINet/train_infer_python.txt 'lite_train_lite_infer'
```

## 7. 致谢
感谢[NAFNet-official](https://github.com/megvii-research/NAFNet)、[MIRNet_paddle](https://github.com/sldyns/MIRNet_paddle)以及[hinet_paddle](https://github.com/youngAt19/hinet_paddle#readme)分享了他们的代码，在本次复现过程中提供了帮助，以及AI Studio提供的算力与答疑支持。
