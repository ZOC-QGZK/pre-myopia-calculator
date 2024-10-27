#Gender [Reference: male]
#INSERT INTO cyc_data_model (model_name, intercept, genddeer_coef, serr_coef, preall_coef, paresdasda_A_coef, paresdasda_B_coef)
#python app.py

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import sqlite3

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# 从数据库中加载模型系数和标准误的函数
def load_model_coefficients(model_name):
    conn = sqlite3.connect('models.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT intercept, intercept_se, 
               genddeer_coef, genddeer_se, 
               variable2_coef, variable2_se,
               preall_coef, preall_se,
               paresdasda_A_coef, paresdasda_A_se,
               paresdasda_B_coef, paresdasda_B_se
        FROM model_coefficients
        WHERE model_name = ?
    ''', (model_name,))

    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            'intercept': row[0],
            'intercept_se': row[1],
            'genddeer_coef': row[2],
            'genddeer_se': row[3],
            'variable2_coef': row[4],
            'variable2_se': row[5],
            'preall_coef': row[6],
            'preall_se': row[7],
            'paresdasda_A_coef': row[8],
            'paresdasda_A_se': row[9],
            'paresdasda_B_coef': row[10],
            'paresdasda_B_se': row[11],
        }
    else:
        raise ValueError("Model coefficients not found for model_name: {}".format(model_name))

# 编码输入的函数
def encode_inputs(inputs):
    # 对 genddeer 进行编码，'male' -> 0, 'female' -> 1
    if inputs['genddeer'].lower() == 'male':
        genddeer = 0
    elif inputs['genddeer'].lower() == 'female':
        genddeer = 1
    else:
        raise ValueError("Invalid value for genddeer: must be 'male' or 'female'")

    # 确保将输入值转换为浮点数
    variable2 = float(inputs['variable2'])
    preall = float(inputs['preall'])

    # paresdasda 处理为数值 0, 1, 2，并进行哑变量编码
    paresdasda = int(inputs['paresdasda'])
    if paresdasda not in [0, 1, 2]:
        raise ValueError("Invalid value for paresdasda: must be 0, 1, or 2")

    # 哑变量编码
    paresdasda_A = 1 if paresdasda == 1 else 0
    paresdasda_B = 1 if paresdasda == 2 else 0

    return {
        "genddeer": genddeer,
        "variable2": variable2,
        "preall": preall,
        "paresdasda_A": paresdasda_A,
        "paresdasda_B": paresdasda_B
    }

# 预测API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 接收请求数据
        data = request.json
        inputs = data.get('inputs')
        group = data.get('group')

        # 检查请求数据的有效性
        if not inputs or not group:
            return jsonify({"error": "Missing required parameters: inputs and group are needed."}), 400

        # 根据组名确定使用的模型
        model_names = []
        if group == 'cyc_data_model':
            model_names = [
                'cyc_data_model_for_1_year',
                'cyc_data_model_for_2_years',
                'cyc_data_model_for_3_years'
            ]
        elif group == 'noncyc_data_model':
            model_names = [
                'noncyc_data_model_for_1_year',
                'noncyc_data_model_for_2_years',
                'noncyc_data_model_for_3_years'
            ]
        else:
            return jsonify({"error": "Invalid group name. Please specify 'cyc_data_model' or 'noncyc_data_model'."}), 400

        results = {}

        for model_name in model_names:
            # 从数据库加载模型系数和标准误
            coefficients = load_model_coefficients(model_name)

            # 编码输入
            encoded_inputs = encode_inputs(inputs)

            # 准备预测数据（包括截距项）
            X = np.array([
                1,  # 截距
                encoded_inputs['genddeer'],
                encoded_inputs['variable2'],
                encoded_inputs['preall'],
                encoded_inputs['paresdasda_A'],
                encoded_inputs['paresdasda_B']
            ])

            # 计算线性组合
            logits = np.dot([
                coefficients['intercept'],
                coefficients['genddeer_coef'],
                coefficients['variable2_coef'],
                coefficients['preall_coef'],
                coefficients['paresdasda_A_coef'],
                coefficients['paresdasda_B_coef']
            ], X)

            # 计算概率
            probability = 1 / (1 + np.exp(-logits))

            # 使用标准误计算置信区间
            # 合并标准误的平方以计算总体的标准误
            variance = (
                coefficients['intercept_se']**2 +
                (coefficients['genddeer_se']**2 * encoded_inputs['genddeer']) +
                (coefficients['variable2_se']**2 * encoded_inputs['variable2']) +
                (coefficients['preall_se']**2 * encoded_inputs['preall']) +
                (coefficients['paresdasda_A_se']**2 * encoded_inputs['paresdasda_A']) +
                (coefficients['paresdasda_B_se']**2 * encoded_inputs['paresdasda_B'])
            )
            standard_error = np.sqrt(variance)

            z = 1.96  # 95% 置信水平的 z 值
            lower_bound_logit = logits - z * standard_error
            upper_bound_logit = logits + z * standard_error

            # 转换置信区间为概率
            lower_bound = 1 / (1 + np.exp(-lower_bound_logit))
            upper_bound = 1 / (1 + np.exp(-upper_bound_logit))

            # 保留小数点后三位
            probability = round(probability, 3)
            lower_bound = round(lower_bound, 3)
            upper_bound = round(upper_bound, 3)

            results[model_name] = {
                "probability": probability,
                "confidence_interval": [lower_bound, upper_bound]
            }

        # 返回所有模型的预测结果
        return jsonify(results)

    except Exception as e:  # 捕获所有异常
        print("Error occurred:", e)  # 打印错误信息到终端
        return jsonify({"error": str(e)}), 500  # 返回500错误及错误信息

if __name__ == '__main__':
    app.run(debug=True)
