# -*- coding: utf-8 -*-
# @USER: CarryChang

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import requests
import pickle
import json

# 设置单个用户评论的最大句子长度
maxlen = 100
# 保存向量字典
tokenize_path = 'model/tokenizer.pickle'

predict_text = ['这个环境不喜欢', '这个环境喜欢不']
# 特征处理
with open(tokenize_path, 'rb') as tokenize_save:
    tokenizer_load = pickle.load(tokenize_save)

# 字符级
tk_list = [list(text) for text in predict_text]
# 字符填充
test_text = pad_sequences(tokenizer_load.texts_to_sequences(tk_list), maxlen)

# 多个评论进行推理
data = {'instances': test_text.tolist()}
# tf_model_textcnn 模型部署，REST 的访问端口为 9501
predict_url = 'http://localhost:9501/v1/models/textcnn:predict'
r = requests.post(predict_url, data=json.dumps(data))
# 直接提取矩阵中积极的情感
print("待测样例的情感值是：")
print(np.array(r.json()['predictions'])[:, 1])
