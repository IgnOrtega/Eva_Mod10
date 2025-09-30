# app.py
import logging
from flask import Flask, request, jsonify
import joblib
import numpy as np

# ==========================================
# 1. Inicializar Flask
# ==========================================
app = Flask(__name__)


# Configurar logging
logging.basicConfig(
    filename='api.log',          # archivo donde se guardarán los logs
    level=logging.INFO,          # nivel de mensajes que se registran
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==========================================
# 2. Cargar modelo serializado
# ==========================================
modelo = joblib.load("./serializar_modelo/Modelo_Breast_Cancer.pkl")

# Número de features esperadas
N_FEATURES = 30

# ==========================================
# 3. Ruta GET para verificar estado
# ==========================================
@app.route("/", methods=["GET"])
def home():
    logging.info("Se accedió a la ruta /")  # registrar un mensaje de info
    return jsonify({"mensaje": "✅ API lista"})

# ==========================================
# 4. Ruta POST para predicción
# ==========================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validar que tenga la clave "features"
        if "features" not in data:
            logging.warning("Falta la clave 'features'")  # warning
            return jsonify({"error": "Falta la clave 'features' en el JSON"}), 400

        features = data["features"]
        logging.info(f"Se recibieron features: {features}")  # info

        # Validar que sea lista de números
        if not isinstance(features, list):
            return jsonify({"error": "'features' debe ser una lista"}), 400
        logging.error(f"Validacion de lista de números completa") 

        if len(features) != N_FEATURES:
            logging.error(f"Cantidad de features distinta a la esperada")
            return jsonify({
                "error": f"Se esperaban {N_FEATURES} valores, pero se recibieron {len(features)}"
            }), 400

        try:
            features = [float(x) for x in features]
        except ValueError:
            return jsonify({"error": "Todos los valores en 'features' deben ser numéricos"}), 400

        # Convertir a array numpy
        features_array = np.array(features).reshape(1, -1)

        # Realizar predicción
        prediction = modelo.predict(features_array)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

# ==========================================
# 5. Ejecutar servidor
# ==========================================
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=6874)


"""
Ejecutar estos código en la powershell:

curl -Method POST http://127.0.0.1:5000/predict `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{"features": [1.93490586,  0.99373946,  1.93309591,  2.01678415,  0.30883801,1.06619158,  2.29003532,  2.1171921 ,  1.43621312, -0.54118592,2.16129581, -0.71857712,  1.735548  ,  2.14655106, -0.58601673,0.76020176,  2.09838476,  1.11014072,  0.41986408,  0.45658573,1.92810613,  0.21540599,  1.72873394,  1.98541627, -0.49396886,0.40035425,  2.04811805,  1.46013593,  0.36439584, -0.30233857]}'
    
"""