## 项目来源

项目来源于[阿里巴巴编程之夏2019]() ，项目的主要内容和目标如[AliOS-Things](https://github.com/alibaba/AliOS-Things/issues/976)所述。简单总结如下：

基于AliOS Things，在设备端MCU上实现一种声纹识别算法，可以根据1段预先录音的PCM格式的录音数据，判断正在录音的声音是否是同一个人的声音。预录的PCM格式的录音时长在10秒左右，要求录音结束后5秒左右可以判断出结果，判断的准确率在80%以上。算法不限，即使用可传统的语音识别，亦可集成某种轻量级AI框架，但要求在设备端MCU上运行，不能将语音数据上传到云端进行判断。

## 项目整体思路

整体的技术方案是在PC端训练AI模型，然后通过轻量级嵌入式AI框架部署到嵌入式系统上。在PC端，训练的数据集是成对的语音处理后的语音频谱，通过CNN模型进行深度Embedding，最后进行相似度比较，得到声纹识别识别的结果。在MCU端，需要录一段或者几段识别人的声音，预录声音会提前进行处理，并保存频谱特征，作为比较的基础，这个过程称为声纹注册。测试的时候说话人录音，经过频谱分析，模型预测得出预测结果，称为声纹识别。

## PC端模型训练和导出

### 说明

**开发环境**：deepin 15.11 桌面版 64位

### 数据集

**THUYG-20**，**下载地址**：http://www.openslr.org/22/，   **License**: Apache License v.2.0

- data_thuyg20.tar.gz [ 2.1G ]（用于语音识别的语音数据和抄本） 
- data_thuyg20_sre.tar.gz [1.6G]（用于说话人识别的语音数据）
- test_noise.tar.gz [773M]（用于语音识别的标准0db噪声测试数据）
- test_noise_sre.tar.gz [1.9G]（标准0db噪声测试数据，用于说话人识别）
- resource.tar.gz [26M]（补充资源，包括训练数据，噪音样本的词典）

### 代码说明

#### 整体代码结构

```python
.
├── LICENSE
├── PC
│   ├── data
│   │   ├── data.py ## 数据处理文件
│   │   ├── test_info.csv ## 数据集信息，样本编号，采样率，时长等
│   │   └── train_info.csv
│   ├── speaker_class_model
│   │   ├── audio_ment
│   │   │   ├── acoustic_guitar_0.wav
│   │   │   ├── demo.py ## 音频数据增强，通过加噪等手段扩充数据集
│   │   │   └── ir
│   │   │       └── impulse_response_0.wav
│   │   ├── nni_model ## nni 参数搜索
│   │   │   ├── config.yml
│   │   │   ├── nni_speaker.py
│   │   │   ├── r_model.py
│   │   │   └── search_space.json
│   │   ├── nnom
│   │   │   ├── fully_connected_opt_weight_generation.py
│   │   │   └── nnom_utils.py
│   │   ├── train_model
│   │   │   ├── mfcc.py  ## MFCC特征提取和转换
│   │   │   ├── python_speech_features ## python 音频特征处理包
│   │   │   │   ├── base.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── sigproc.py
│   │   │   ├── run.py ## 主要的模型训练脚本
│   │   │   └── test.py ## 测试分类模型和Embedding模型
│   │   └── WebApp ## WEB 页面
│   │       ├── app
│   │       │   ├── app.py
│   │       │   ├── app.pyc
│   │       │   ├── config.py
│   │       │   ├── __init__.py
│   │       │   ├── __init__.pyc
│   │       │   ├── kws.py
│   │       │   ├── mfcc.py
│   │       │   ├── static
│   │       │   │   ├── recorder.js
│   │       │   │   └── style.css
│   │       │   ├── templates
│   │       │   │   ├── enroll_speaker.html
│   │       │   │   ├── error.html
│   │       │   │   ├── index.html
│   │       │   │   ├── layout.html
│   │       │   │   └── recognize_speaker.html
│   │       │   ├── test.py
│   │       │   └── utils.py
│   │       ├── README.md
│   │       ├── setup.py
│   │       ├── test_data
│   │       │   ├── voices_processed
│   │       │   └── voices_raw
│   │       └── train_data
│   │           ├── voices_processed
│   │           └── voices_raw
│   └── triplet-loss #triplet-loss model
│       ├── bases
│       │   ├── data_loader_base.py
│       │   ├── infer_base.py
│       │   ├── __init__.py
│       │   ├── model_base.py
│       │   └── trainer_base.py
│       ├── configs
│       │   └── triplet_config.json
│       ├── data_loaders
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── __init__.cpython-36.pyc
│       │   │   └── triplet_dl.cpython-36.pyc
│       │   └── triplet_dl.py
│       ├── infers
│       │   ├── __init__.py
│       │   └── triplet_infer.py
│       ├── main_train.py
│       ├── models
│       │   ├── __init__.py
│       │   └── triplet_model.py
│       ├── print_state.py
│       ├── root_dir.py
│       ├── trainers
│       │   ├── __init__.py
│       │   └── triplet_trainer.py
│       └── utils
│           ├── config_utils.py
│           ├── __init__.py
│           ├── np_utils.py
│           └── utils.py
└── README.md
```

#### 数据处理

对**训练集**音频数据（16kHZ）按时间（2s）切分为训练数据，提取MFCC特征，用于模型识别speaker。

对**测试者**，也对音频进行切分，可以每个音频都进行识别，然后根据识别结果取最多的，相应的当音频时间较长的时候时间成本会增加。本项目只用前2s的数据进行识别。

![](https://user-gold-cdn.xitu.io/2019/7/12/16be4210b28fdfb9?w=640&h=480&f=png&s=16080)

#### Speaker分类模型

分类模型的优点是准确率高，缺点也很明显就是每当加入新的speaker，模型需要联网更新，而且模型体积会相应变大，也就是对speaker的数据有上限要求。

#### Embedding模型

分别测试了Deep Vector论文里面的模型和Triplet Loss模型，均表现不好。

- Deep Vector 最后测试的Top3的准确率为50%左右
- Triplet Loss模型模型很难收敛，原因是有的类别数量较少，导致总的模型训练集数据比较少。

#### NNI模型参数搜索

![nA0uAe.png](https://s2.ax1x.com/2019/09/03/nA0uAe.png)

最好的参数如下：

```json
{
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "batch_size": 128,
    "dropout_rate": 0.2778256049814192
}
```

#### WEB页面展示

**整体**页面展示

![nED22d.png](https://s2.ax1x.com/2019/09/04/nED22d.png)

**注册**页面展示

![nEDbGQ.png](https://s2.ax1x.com/2019/09/04/nEDbGQ.png)



**模型训练**页面展示

![nErSaT.png](https://s2.ax1x.com/2019/09/04/nErSaT.png)

**声纹识别**页面展示

![nErVqx.png](https://s2.ax1x.com/2019/09/04/nErVqx.png)

模型WEB页面演示视频：[B站](https://www.bilibili.com/video/av66574342/)

#### 模型量化和导出

利用nnom生成weights.h，生成代码如下：

```python
generate_test_bin(x_train * 127, y_train, 'train_data.bin')
generate_model(model, x_train, name="weights.h")
```

## 嵌入式端模型部署

使用的是[AliOS Things 2.1](https://github.com/alibaba/AliOS-Things)版本，板子型号为STM32L476VGT64。

![nVlLy4.png](https://s2.ax1x.com/2019/09/04/nVlLy4.png)

官方没有这个板子的代码，所以我自己修改了一下，具体修改参考：

- [Alios Things LED闪烁](https://juejin.im/post/5d32fd3ce51d45108c59a61c)

嵌入式板子上的[演示视频](https://www.bilibili.com/video/av66589733)

## TO DO

- 更多的代码注释
- 优化项目代码结构
- 提高分类模型的准确率



## 鸣谢

- [NNI](https://github.com/microsoft/nni)
- [NNOM](https://github.com/majianjia/nnom)
- [SpeakerRecognition](https://github.com/rjagiasi/SpeakerRecognition)

 

