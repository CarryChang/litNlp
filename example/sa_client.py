#! -*- coding: utf-8 -*-
import requests
import time
import json


def sa_api_request(content):
    st = time.time()
    api_url = 'http://127.0.0.1:5021/sa_api'
    para = {"content": content}
    model_result = requests.post(api_url, data=json.dumps(para)).json()
    print(model_result)
    print('request time used:{}'.format(time.time() - st))


if __name__ == '__main__':
    content = '这家酒店真的不错'
    # 接口请求
    sa_api_request(content)
