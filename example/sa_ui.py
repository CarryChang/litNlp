# -*- coding: utf-8 -*-
from litNlp.predict import SA_Model_Predict
import streamlit as st
# 初始化模型
tokenize_path = 'model/tokenizer.pickle'
sa_model_path_m = 'model/model.h5'
model = SA_Model_Predict(tokenize_path, sa_model_path_m, max_len=100)
# 不用项目自动重启
st.subheader('文本情感分析')
# st.write('文本情感分析')
# 接受前端的内容显示
comment_input = st.text_input('请输入一行测试文本: ')
# 开始处理内容
if comment_input != '':
    # 文本处理
    comment = str(comment_input).strip()
    # 添加等待，并开始预测
    with st.spinner('Predicting...'):
        sa_score = float(model.predict([comment])[0][1])
        show_data = dict()
        if sa_score > 0.5:
            show_data['label'] = '积极'
            show_data['sa_score'] = sa_score
        elif sa_score < 0.5:
            show_data['label'] = '消极'
            show_data['sa_score'] = sa_score
        else:
            show_data['label'] = '中性'
            show_data['sa_score'] = sa_score
        show_data['status'] = 1
    # 最后展示内容
    st.write('分析结果: ')
    st.write(show_data)