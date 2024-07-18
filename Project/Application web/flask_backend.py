from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)

# # Cargar el modelo ML
try: 
    with open('predictor_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: El archivo 'predictor_model.pkl' no se encontró.")
    model = None

@app.route('/')
def serve_html():
    return send_from_directory(os.path.dirname(__file__), 'Heart_Disease.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Modelo no cargado correctamente'}), 500

    try:
        data = request.json
        # Preparar los datos para la predicción
        features = [
            data['chestpain'],
            data['restingBP'],
            data['serumcholestrol'],
            data['fastingbloodsugar'],
            data['restingelectro'],
            data['maxheartrate'],
            data['slope'],
            data['noofmajorvessels']
        ]
        # Convertir las características a un array numpy y asegurarse de que todos los valores sean numéricos
        features = np.array(features).astype(float).reshape(1, -1)
        # Realizar la predicción
        prediction = model.predict(features)[0]
        # Devolver la predicción como respuesta JSON
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error':str(e)}),400
    
if __name__ == '__main__':
    app.run(debug=True)
