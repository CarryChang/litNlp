# -*- coding: utf-8 -*-
# @Time: 2020/6/20 0020 10:38
from tensorflow.keras.layers import Dense, Embedding, Flatten,Dropout,Convolution1D,BatchNormalization
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional,Activation,LSTM
class BILSTM_Model:
    def create_model(self, max_words,embedding_dim, maxlen, n_class=2):
        model = Sequential()
        #  embedding layer
        model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
        # BiLSTM
        model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
        model.add(LSTM(units=16, return_sequences=False))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_class,  activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
        return model