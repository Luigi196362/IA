import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__)

# Configuración
MODEL_PATH = 'modelo_entrenado.keras'
LABELS_PATH = 'labels.txt'
IMG_SIZE = (224, 224)

# Cargar modelo y etiquetas
print("Cargando modelo...")
try:
    model = load_model(MODEL_PATH)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

print("Cargando etiquetas...")
try:
    with open(LABELS_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"Etiquetas cargadas: {class_names}")
except Exception as e:
    print(f"Error al cargar etiquetas: {e}")
    class_names = []

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'El modelo no está cargado.'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró la parte de la imagen'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ninguna imagen'}), 400

    if file:
        try:
            # Guardar temporalmente la imagen
            temp_path = os.path.join('temp_img.jpg')
            file.save(temp_path)

            # Preprocesamiento
            img = load_img(temp_path, target_size=IMG_SIZE)
            img_array = img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Crear un lote (batch)

            # Predicción
            predictions = model.predict(img_array)
            score = predictions[0]
            
            predicted_class = class_names[np.argmax(score)]
            confidence = 100 * np.max(score)

            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)

            return jsonify({
                'class': predicted_class,
                'confidence': float(confidence)
            })
        except Exception as e:
            # Asegurar limpieza en caso de error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
