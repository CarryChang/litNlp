# -*- coding: utf-8 -*-
# @Time: 2020/6/20 0020 0:55
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 解决跨文件问题
tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'sa_model/tokenizer.pickle')
sa_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sa_model/c_cnn_m.h5')
class SA_Model_Predict:
    def __init__(self,tokenize_path=tokenizer_path):
        with open(tokenize_path, 'rb') as tokenize_save:
            self.tokenizer_load = pickle.load(tokenize_save)
    def predict(self,predict_text,maxlen=100,sa_model_path_m= sa_model_path):
        tk_list = [list(text) for text in predict_text]
        test_text = pad_sequences(self.tokenizer_load.texts_to_sequences(tk_list), maxlen)
        model_load = load_model(sa_model_path_m)
        test_proba_list = model_load.predict(test_text)
        return test_proba_list
if __name__ == '__main__':
    # 内置参数，批处理文本
    predict_text = ['这个我不喜欢', '这个我喜欢不']
    # 初始化模型
    model = SA_Model_Predict()
    sa_score = model.predict(predict_text)
    # 多分类模型输出
    print([i[1] for i in sa_score])