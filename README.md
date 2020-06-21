 
# litNlp
## 简介
litNlp是基于Tensorflow2.0实现的一个轻量级的深度文本分类模型,支持多分类，并默认二分类，是搭建文本分类模型的快速方案。
1. 内置情感分类模型

# 使用方法
>pip install  litNlp

    from litNlp.predict import SA_Model_Predict
    # 批处理文本
    predict_text = ['这个我不喜欢', '这个我喜欢不']
    # 初始化并加载模型
    tokenizer_path = 'sa_model/tokenizer.pickle'
    sa_model_path = 'sa_model/c_cnn_m.h5'
    model = SA_Model_Predict(tokenizer_path)
    sa_score = model.predict(predict_text,sa_model_path_m=sa_model_path)
    print([i[1] for i in sa_score])
