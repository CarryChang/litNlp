# -*- coding: utf-8 -*-
# @Time: 2020/6/21 0021 20:46
import pandas as pd
from litNlp.train import SA_Model_Train
# e_comment
# train_data = pd.read_csv('data/ebusiness_comment.csv')
# hotel
train_data = pd.read_csv('data/hotel_comment.csv')
# data processs
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
# train_method : 模型训练方式，默认textcnn，可选：bilstm , gru
train_method = 'bilstm'
# train: evaluate默认在训练完毕之后开启计算
label = train_data['label']
train_data = train_data['text_cut']
model = SA_Model_Train(max_words, embedding_dim, maxlen, tokenize_path, sa_model_path_m, train_method)
model.train(train_data, label, num_classes, batch_size=256, epochs=10, verbose=1, evaluate=True)