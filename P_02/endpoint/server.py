
from my_utilities import DataSetPreparation, feature_engineer, PredictData, load_model
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Carga el modelo al inicio
pipeline = load_model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data['instances'])
    transformed_data = pipeline.transform(df)
    return jsonify(transformed_data.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

