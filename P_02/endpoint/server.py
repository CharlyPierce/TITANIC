
from my_utilities import DataSetPreparation, feature_engineer, PredictData, load_model
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Carga el modelo al inicio
pipeline = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['instances'])
        transformed_data = pipeline.transform(df)
        return jsonify({"predictions": transformed_data.tolist()})
    except Exception as e:
        # Es útil registrar errores para facilitar la depuración.
        app.logger.error(f"Error al hacer la predicción: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

