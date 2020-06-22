# -*- coding: utf-8 -*-
# @Time: 2020/6/21 0021 20:46
import pandas as pd
from litNlp.train import SA_Model_Train
train_data = pd.read_csv('data/e_comment.csv')
# data process
train_data['text_cut'] = train_data['text'].apply(lambda x: " ".join(list(x)))
# 最大句子长度
maxlen = 100
# 最大的tokenizer字典长度
max_words = 1000
# 设置embedding大小
embedding_dim = 300
# 模型的保存位置，后续用于推理
sa_model_path_m = 'model/model.h5'
# 离线保存tokenizer
tokenize_path ='model/tokenizer.pickle'
# 分类的类别数
num_classes = 2
label = train_data['label']
train_data = train_data['text_cut']
# train: evaluate默认在训练完毕之后开启计算
model = SA_Model_Train(max_words, embedding_dim, maxlen, tokenize_path, sa_model_path_m)
model.train(train_data, label, num_classes, batch_size=256, epochs=2, verbose=1, evaluate=True)