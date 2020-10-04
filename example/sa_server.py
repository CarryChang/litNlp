#! -*- coding: utf-8 -*-
from flask_restful import Resource, Api, request
from litNlp.predict import SA_Model_Predict
from flask import Flask
import json

app = Flask(__name__)
api = Api(app)

# 初始化模型，第一次推理需要预热
tokenize_path = 'model/tokenizer.pickle'
sa_model_path_m = 'model/model.h5'
# 模型加载
model = SA_Model_Predict(tokenize_path, sa_model_path_m, max_len=100)


class sa_post_api(Resource):
    def post(self):
        # 接收对象
        parser = json.loads(request.get_data())
        content = str(parser['content'])
        sa_score = round(float(model.predict([content])[0][1]), 5)
        show_data = dict()
        show_data['sa_score'] = sa_score
        show_data['status'] = 1
        if sa_score > 0.5:
            show_data['label'] = '积极'
        elif sa_score < 0.5:
            show_data['label'] = '消极'
        else:
            show_data['label'] = '中性'
        # print(show_data)
        return show_data


# 定义 POST 接口的请求信息
api.add_resource(sa_post_api, '/sa_api')

if __name__ == '__main__':
    app.run(port='5021')
