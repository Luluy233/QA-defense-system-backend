
from .predict_harmful import predict_harm
from .predict_similarity import similarity

from flask import (
    Blueprint, request, jsonify
)
from flask_cors import CORS

bp = Blueprint('classify', __name__, url_prefix='/classify')
CORS(bp)


@bp.route('/similarity', methods=["GET"])
def is_similarity():
    try:
        text1 = request.args.get('text1', default='', type=str)
        text2 = request.args.get('text2', default='', type=str)
        result = 0
        if(text1!=''):
            result = similarity(text1,text2)
          
        print(result)
        response_data = {
            'code': 200,
            'data': {
                'tag': result
            }
        }
        return jsonify(response_data)

    except Exception as e:
        # 处理其他异常
        response_data = {
            'code': 500,
            'error': 'An internal server error occurred.'
        }
        return jsonify(response_data), 500


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






