import json
import keras
from flask import Flask
from flask import request
import tensorflow as tf
from src.utils import config
from src.ML.models import Models

# 下面三行没有什么作用，只是为了保证keras正常不报错
global graph, sess
graph = tf.get_default_graph()
sess = keras.backend.get_session()

# 初始化模型， 避免在函数内部初始化，耗时过长
model = Models(model_path=config.root_path + '/model/ml_model/lightgbm', train_mode=False)

app = Flask(__name__)

@app.route('/predict', methods=["POST"])
def gen_ans():
    '''
    以RESTful的方式获取模型结果, 传入参数为title:图书标题，desc:图书描述
    返回json格式，包含标签和对应概率
    '''
    result = {}
    title = request.form['title']
    desc = request.form['desc']
    with sess.as_default():
        with graph.as_default():
            label, score = model.predict(title, desc)
    result = {
        "label": label,
        "proba": str(score)
    }
    return json.dumps(result, ensure_ascii=False)


# python3 -m flask run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)