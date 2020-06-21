# -*- coding: utf-8 -*-
# @Time: 2020/6/20 0020 0:55
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
class SA_Model_Predict:
    def __init__(self,tokenize_path, sa_model_path_m, max_len=100):
        with open(tokenize_path, 'rb') as tokenize_save:
            self.tokenizer_load = pickle.load(tokenize_save)
        self.max_len = max_len
        self.sa_model_path_m = sa_model_path_m
    def predict(self,predict_text):
        tk_list = [list(text) for text in predict_text]
        test_text = pad_sequences(self.tokenizer_load.texts_to_sequences(tk_list), self.max_len)
        model_load = load_model(self.sa_model_path_m)
        test_proba_list = model_load.predict(test_text)
        return test_proba_list
# if __name__ == '__main__':
#     # 内置参数，批处理文本
#     predict_text = ['这个我不喜欢', '这个我喜欢不']
#     # 初始化模型
#     model = SA_Model_Predict(tokenize_path=tokenizer_path, sa_model_path_m=sa_model_path)
#     sa_score = model.predict(predict_text)
#     # 多分类模型输出
#     print([i[1] for i in sa_score])