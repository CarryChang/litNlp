<div align="center">
  <img src="https://github.com/CarryChang/litNlp/blob/master/pic/logo.png"><br>
</div>

-----------------
## litNlp: A Fast Tool for Sentiment Analysis with Tensorflow2
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)
[![PyPI Latest Release](https://img.shields.io/pypi/v/litNlp.svg)](https://pypi.org/project/litNlp/)
[![Downloads](https://pepy.tech/badge/litnlp)](https://pepy.tech/project/litnlp)
[![Downloads](https://pepy.tech/badge/litnlp/month)](https://pepy.tech/project/litnlp/month)
[![Downloads](https://pepy.tech/badge/litnlp/week)](https://pepy.tech/project/litnlp/week)


# litNlp 简介

litNlp 是兼容最新版 Tensorflow 2.0 实现的一个轻量级的深度情感极性推理模型，使用字符级代替词语级进一步提升训练和推理速度，可以实现细粒度的多级别情感极性训练和预测，TF2 下 GPU 和 CPU 平台都能直接安装运行，是搭建 NLP 情感分析和分类模型 Baseline 的快速方案。

	1. 内置深度学习情感分析模型。
	2. 直接提供模型训练，默认 Text-CNN 字符级卷积网络作为 baseline ，自带早停操作，使用少的参数即可开始训练多分类模型。
	3. 使用 Streamlit 快速对模型进行 UI 演示。
	4. 增加 TF Serving 的转化和部署。
	5. 增加 docker-compose up 的启动方式


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
	import numpy as np
	
	# 加载模型的字典项
	tokenize_path = 'model/tokenizer.pickle'
	# train_method : 模型训练方式，默认 textcnn ，可选：bilstm , gru
	train_method = 'textcnn'
	# 模型的保存位置，后续用于推理
	sa_model_path_m = 'model/{}.h5'.format(train_method)
	# 开始输入待测样例
	predict_text = ['这个我不喜欢', '这个我喜欢不']
	# 加载模型
	model = SA_Model_Predict(tokenize_path, sa_model_path_m, max_len=100)
	# 开始推理
	sa_score = model.predict(predict_text)
	# 情感极性概率
	print(np.asarray(sa_score)[:,1])
	# 情感label输出
	print(np.argmax(np.asarray(sa_score), axis=1))

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

## jupyter 实验

>  情感分析，优化语义的情感推理
<div align=center><img  src="https://github.com/CarryChang/litNlp/blob/master/pic/tools.png"></div>

## Flask Gunicorn 模型部署
python sa_server.py 即可对训练的情感分析模型进行部署，模型首次推理需要预热，后续推理耗时在 200ms 之内。

<div align=center><img  src="https://github.com/CarryChang/litNlp/blob/master/pic/server.png"></div>

## Tensorflow Serving 模型部署

利用 python example/sa_model2tf_serving_model.py 进行模型转换之后即可直接进行 TF Serving 的服务部署。

首先拉取对应版本的 TF Serving Docker

    docker pull tensorflow/serving:2.3.0
    
直接利用 Docker 加载转换之后的模型即可完成模型部署，TensorFlow Serving 会自动选择版本号最大的模型进行载入。

Docker 命令行的 Dev 启动模式

    docker run -t --rm -p 9500:8500 -p:9501:8501 \
    -v "$(pwd)/tf_model/:/models/textcnn" \
    -e MODEL_NAME=textcnn -tensorflow_inter_op_parallelism=4 \
    tensorflow/serving:2.3.0
 

Docker 命令行的 Pro 启动模式

    docker run -d --rm -p 9500:8500 -p:9501:8501 \
    -v "$(pwd)/tf_model/:/models/textcnn" \
    -e MODEL_NAME=textcnn -tensorflow_inter_op_parallelism=4 \
    tensorflow/serving:2.3.0

或者在 yml 所在的文件夹下增加直接使用 docker-compose up 进行服务的启动。

服务请求： 
部署之后使用 python sa_tf_serving_api_client.py 进行 TF serving 服务的调用。
 
