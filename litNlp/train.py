# -*- coding: utf-8 -*-
# @Time: 2020/6/20 0020 0:46
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from .model_structure.TextCNN_m import TextCNN_m
from sklearn import metrics
import numpy as np
import pickle
class SA_Model_Train:
    def __init__(self, max_words, embedding_dim,maxlen,tokenize_path,sa_model_path_m):
        self.init_model = TextCNN_m()
        self.max_words = max_words
        self.tokenize_path = tokenize_path
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen
        self.sa_model_path_m = sa_model_path_m
        self.model = self.init_model.create_model(self.max_words,self.embedding_dim, self.maxlen)
    def train_tk(self,train_data):
        tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', num_words=self.max_words)
        tokenizer.fit_on_texts(train_data)
        with open(self.tokenize_path, 'wb') as tokenize:
            pickle.dump(tokenizer, tokenize)
        return tokenizer
    def train(self,train_data,label,num_classes,batch_size=256,epochs=10,verbose=1,evaluate=True):
        # to_categorical
        targets_values = to_categorical(label, num_classes=num_classes)
        # data split
        x_train, y_train, x_test, y_test = train_test_split(train_data, targets_values, test_size=0.2, random_state=1)
        # pad_sequences
        tokenizer = self.train_tk(train_data)
        x_train, x_test = pad_sequences(tokenizer.texts_to_sequences(x_train), self.maxlen), np.array(x_test)
        y_train, y_test = pad_sequences(tokenizer.texts_to_sequences(y_train), self.maxlen), np.array(y_test)
        self.model.fit(x_train, x_test, batch_size, epochs, verbose,
                       validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])
        self.model.save(self.sa_model_path_m)
        try:
            pre_result = self.model.predict(y_train, batch_size=256, verbose=0)
        except:
            result_ = self.model.predict_classes(y_train, batch_size=256, verbose=0)
            pre_result = np.argmax(result_, axis=1)
        if evaluate:
            result = [np.argmax(i) for i in pre_result]
            y_test = [np.argmax(i) for i in y_test]
            report = metrics.classification_report(y_test, result)
            acc = metrics.accuracy_score(y_test, result)
            auc = metrics.roc_auc_score(y_test, result)
            print(report)
            print('acc:  {}    auc:  {}'.format(acc, auc))
# if __name__ == '__main__':
#     # C-CNN-SA(字符级卷积网络)
#     train_data = pd.read_csv('data/sa_data_train.csv')
#     # list sentence
#     train_data['text_cut'] = train_data['text'].apply(lambda x: " ".join(list(x)))
#     model = SA()
#     tokenizer = model.train_tk()
#     # 2-8的分割数据,固定测试数据
#     targets_values = to_categorical(train_data['label'], num_classes=num_classes)
#     x_train, y_train, x_test, y_test = train_test_split(train_data['text_cut'],targets_values, test_size=0.2, random_state=1)
#     # pad_sequences
#     x_train, x_test = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen), np.array(x_test)
#     y_train, y_test = pad_sequences(tokenizer.texts_to_sequences(y_train), maxlen), np.array(y_test)
#     # train
#     pre_result = model.train(x_train, x_test)
#     # evaluate
#     model.evaluate(pre_result, y_test)






