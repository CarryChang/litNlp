[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)
[![Downloads](https://pepy.tech/badge/litnlp)](https://pepy.tech/project/litnlp)
[![Downloads](https://pepy.tech/badge/litnlp/month)](https://pepy.tech/project/litnlp/month)
[![Downloads](https://pepy.tech/badge/litnlp/week)](https://pepy.tech/project/litnlp/week)


# litNlp 简介

litNlp 是基于 Tensorflow 2.0 实现的一个轻量级的深度情感极性推理模型，使用字符级代替词语级进一步提升训练和推理速度，可以实现细粒度的多级别情感极性训练和预测。

	TF2 下 GPU 和 CPU 平台都能直接安装运行，是搭建 NLP 分类模型类 baseline 的快速方案。

	1. 内置情感分析模型-利用深度模型优化语义建模，使用字符级减少 tokenizer 的大小。
	2. 直接提供模型训练，默认 Text-CNN 字符级卷积网络作为 baseline ，自带早停操作，使用少的参数即可开始训练多分类模型。

## 直接使用 emample/sa_ui.py 进行前端 ui 展示效果

```python
    # 安装 streamlit 之后直接运行脚本
    streamlit run sa_ui.py
```
<div align=center><img  src="https://github.com/CarryChang/litNlp/blob/master/pic/ui.png"></div>

## 使用方法
> 1. pip install  litNlp
> 2. 模型需要先通过训练，保存在 sa_model 里面，然后就可以批预测，具体的使用见 example 文件内容

```python
    from litNlp.predict import SA_Model_Predict
    # 批处理文本
    predict_text = ['这个我不喜欢', '这个我喜欢不']
    # 初始化并加载模型
    tokenizer_path = 'sa_model/tokenizer.pickle'
    sa_model_path = 'sa_model/c_cnn_m.h5'
    model = SA_Model_Predict(tokenizer_path,sa_model_path_m=sa_model_path)
    sa_score = model.predict(predict_text)
    print([i[1] for i in sa_score])
```

## 参数解释
```python
    # 最大句子长度
    maxlen = 100
    # 最大的tokenizer字典长度
    max_words = 1000
    # 设置embedding大小
    embedding_dim = 300
    # 模型的保存位置，后续用于推理
    sa_model_path_m = 'sa_model/c_cnn_m.h5'
    # 离线保存tokenizer
    tokenize_path ='sa_model/tokenizer.pickle'
    # 分类的类别数
    num_classes = 2
    # train_method : 模型训练方式，默认textcnn，可选：bilstm, gru
    train_method = 'textcnn'
```

## 2 个 epoch 的二分类性能

<div align=center><img  src="https://github.com/CarryChang/litNlp/blob/master/pic/auc_2poch.png"></div>

## jupyter 加载
>  情感分析，优化语义的情感推理
<div align=center><img  src="https://github.com/CarryChang/litNlp/blob/master/pic/tools.png"></div>
 
### [pipy code](https://pypi.org/project/litNlp/)