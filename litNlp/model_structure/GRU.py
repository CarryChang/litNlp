# -*- coding: utf-8 -*-
# @Time: 2020/6/20 0020 10:38
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Embedding, Flatten,Dropout,Convolution1D,BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import concatenate,GlobalAveragePooling1D,Activation,MaxPool1D,Input,GRU
class GRU_Model:
    def create_model(self,max_words,embedding_dim,maxlen):
        model = Sequential()
        #  embedding layer
        model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
        # GRU
        model.add(GRU(units=32, return_sequences=True))
        model.add(GRU(units=16, return_sequences=False))
        model.add(Dense(1,  activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
        return model
# if __name__ == '__main__':
#     model = sa_model()
#     sa = model.create_model()
#     sa.summary()