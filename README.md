 
# litNlp
## 简介
litNlp是基于Tensorflow2.0实现的一个轻量级的深度文本分类模型,支持多分类，并默认二分类，是搭建文本分类模型的快速方案。
1. 内置情感分类模型-利用深度模型优化语义建模，使用字符级减少tokenizer的大小

# 使用方法
> 1. pip install  litNlp
> 2. 模型保存在litNlp的sa_model里面，需要下载之后进行加载

    from litNlp.predict import SA_Model_Predict
    # 批处理文本
    predict_text = ['这个我不喜欢', '这个我喜欢不']
    # 初始化并加载模型
    tokenizer_path = 'sa_model/tokenizer.pickle'
    sa_model_path = 'sa_model/c_cnn_m.h5'
    model = SA_Model_Predict(tokenizer_path,sa_model_path_m=sa_model_path)
    sa_score = model.predict(predict_text)
    print([i[1] for i in sa_score])
    
# jupyter 加载
> 1. 通过高频词可视化展示，归纳出评论主题
<div align=center><img  src="https://github.com/CarryChang/litNlp/blob/master/pic/tools.png"></div>
 