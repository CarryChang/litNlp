# -*- coding: utf-8 -*-

import tensorflow as tf


# 待转化的模型，默认 textcnn ，可选：bilstm , gru
train_method = 'textcnn'
# 模型的保存位置，后续用于推理
sa_model_path_m = 'model/{}.h5'.format(train_method)
# 模型加载
model = tf.keras.models.load_model(sa_model_path_m)
# TF Serving 按照最大的 tag 进行模型的热更新，设置模型的tag
tag = 1
# 转化之后的模型路径
save_path = "tf_model/{}/".format(tag)
# 保存为 tf serving 加载的 model 形式
model.save(save_path, save_format='tf') 

