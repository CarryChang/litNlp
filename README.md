[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

# litNlp
## 简介
litNlp是基于Tensorflow2.0实现的一个轻量级的深度文本分类模型,类别默认2分类，
GPU和CPU平台通用，是搭建文本分类模型baseline的快速方案。
1. 内置情感分析模型-利用深度模型优化语义建模，使用字符级减少tokenizer的大小
2. 直接提供模型训练，默认Text-CNN字符级卷积网络作为baseline，自带早停操作，使用少的参数即可开始训练多分类模型

# 使用方法
> 1. pip install  litNlp
> 2. 模型需要先通过训练，保存在sa_model里面，然后就可以批预测，具体的使用见example文件内容

    from litNlp.predict import SA_Model_Predict
    # 批处理文本
    predict_text = ['这个我不喜欢', '这个我喜欢不']
    # 初始化并加载模型
    tokenizer_path = 'sa_model/tokenizer.pickle'
    sa_model_path = 'sa_model/c_cnn_m.h5'
    model = SA_Model_Predict(tokenizer_path,sa_model_path_m=sa_model_path)
    sa_score = model.predict(predict_text)
    print([i[1] for i in sa_score])
 
# 参数解释
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
# 10的epoch 的二分类性能
              precision    recall  f1-score   support

           0       0.94      0.90      0.92      3758
           1       0.89      0.93      0.91      3218

    accuracy                           0.92      6976
    macro avg       0.91      0.92      0.92      6976
    weighted avg       0.92      0.92      0.92      6976
    acc:  0.9157110091743119    auc:  0.9166987782599937

# jupyter 加载
> 1. 情感分析，优化语义的情感推理
<div align=center><img  src="https://github.com/CarryChang/litNlp/blob/master/pic/tools.png"></div>
<div align=center><img  src="https://github.com/CarryChang/litNlp/blob/master/pic/auc.png"></div>
 
### [pipy code](https://pypi.org/project/litNlp/)