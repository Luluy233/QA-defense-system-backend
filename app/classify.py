
from .predict_harmful import predict_harm

from flask import (
    Blueprint, request, jsonify
)

bp = Blueprint('classify', __name__, url_prefix='/classify')


@bp.route('/similarity', methods=["POST"])
def is_similarity():
    text1 = request.form['text1']
    text2 = request.form['text2']


@bp.route('/harmful', methods=['GET'])
def is_harmful():
    # import os
    # current_directory = os.getcwd()
    # print("当前工作目录是：", current_directory)

    try:
        text = request.args.get('text', default='', type=str)
        print(text)
        result = predict_harm('best_lr=2e-6_epochs=3.pt', text)
        print(result)
        response_data = {
            'code': 200,
            'data': {
                'tag': result
            }
        }
        return jsonify(response_data)
    except Exception as e:
        print(e)
        # 处理其他异常
        response_data = {
            'code': 500,
            'error': 'An internal server error occurred.'
        }
        return jsonify(response_data), 500






