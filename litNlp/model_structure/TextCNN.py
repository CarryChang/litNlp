# -*- coding: utf-8 -*-
# @Time: 2020/6/20 0020 10:38
from tensorflow.keras.layers import Dense, Embedding, Flatten,Dropout,Convolution1D,BatchNormalization
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import concatenate,GlobalAveragePooling1D,Activation,MaxPooling1D,Input
class sa_model:
    def create_model(self, max_words, embedding_dim, maxlen):
        model = Sequential()
        #  embedding layer
        model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
        model.add(Convolution1D(64, 3, input_shape=(-1, embedding_dim)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2, 2))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1,  activation='sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-3), metrics=['accuracy'])
        return model
# if __name__ == '__main__':
#     model = sa_model()
#     sa = model.create_model()
#     sa.summary()