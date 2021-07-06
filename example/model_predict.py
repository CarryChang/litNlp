# -*- coding: utf-8 -*-
 

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