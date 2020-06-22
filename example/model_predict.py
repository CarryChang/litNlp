# -*- coding: utf-8 -*-
# @Time: 2020/6/21 0021 14:07
from litNlp.predict import SA_Model_Predict
# 内置参数，批处理文本
predict_text = ['这个我不喜欢', '这个我喜欢不']
# 初始化模型
# 解决跨文件问题
tokenize_path = 'sa_model/tokenizer.pickle'
sa_model_path_m = 'sa_model/c_cnn_m.h5'
model = SA_Model_Predict(tokenize_path, sa_model_path_m, max_len=100)
sa_score = model.predict(predict_text)
# 多分类模型输出
print([i[1] for i in sa_score])