import requests
import os

url = 'http://127.0.0.1:5000/predict'
# Usar una imagen existente
image_path = 'Dataset/Autos/Autos_01.jpg'

if not os.path.exists(image_path):
    print(f"Error: La imagen {image_path} no existe.")
else:
    try:
        print(f"Enviando imagen: {image_path}")
        files = {'image': open(image_path, 'rb')}
        response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error Response: {response.text}")
    except Exception as e:
        print(f"Error de conexión: {e}")
