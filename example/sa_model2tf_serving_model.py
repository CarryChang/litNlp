# -*- coding: utf-8 -*-

import tensorflow as tf

# model2tfserving
model = tf.keras.models.load_model(sa_model_path_m)
save_path = "tf_model/1/"
# 保存为 tf serving 加载的 model 形式
model.save(save_path, save_format='tf')  # 导出tf格式的模型文件

