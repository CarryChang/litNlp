# -*- coding: utf-8 -*-
# @Time: 2020/6/21 0021 20:46
import pandas as pd
from litNlp.train import SA_Model_Train
# e_comment
# train_data = pd.read_csv('data/ebusiness_comment.csv')
# hotel
train_data = pd.read_csv('data/hotel_comment.csv')
# 进行字符级处理
train_data['text_cut'] = train_data['text'].apply(lambda x: " ".join(list(x)))
# 最大句子长度
maxlen = 100
# 设置 tokenizer 字典大小
max_words = 1000
# 设置随机 embedding 大小
embedding_dim = 300
# train_method : 模型训练方式，默认 textcnn ，可选：bilstm , gru
train_method = 'textcnn'
# 模型的保存位置，后续用于推理
sa_model_path_m = 'model/{}.h5'.format(train_method)
# 离线保存 tokenizer
tokenize_path ='model/tokenizer.pickle'
# train: evaluate默认在训练完毕之后开启计算
label = train_data['label']
train_data = train_data['text_cut']
model = SA_Model_Train(max_words, embedding_dim, maxlen, tokenize_path, sa_model_path_m, train_method)
# 模型使用两极情感标注，定义 2 类标签类别，参数可以调节
model.train(train_data, label, num_classes=2, batch_size=256, epochs=2, verbose=1, evaluate=True)