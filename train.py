import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

# --- Configuración ---
# Definir rutas y parámetros
dataset_path = 'Dataset'
data_dir = pathlib.Path(dataset_path)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
EPOCHS = 5

print(f"Usando directorio de datos: {data_dir.resolve()}")

# --- Cargar Datos ---
# Usamos image_dataset_from_directory para cargar las imágenes eficientemente
# Se divide en entrenamiento (80%) y validación (20%)
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

class_names = train_ds.class_names
print(f"Clases encontradas: {class_names}")
num_classes = len(class_names)

# Optimización de rendimiento
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Data Augmentation ---
# Esto genera variaciones de las imágenes para mejorar el entrenamiento con pocos datos
data_augmentation = Sequential([
  layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
  layers.RandomContrast(0.2),
])

# --- Modelo ---
# Usamos MobileNetV2 pre-entrenado como base (Transfer Learning)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Modelo base pre-entrenado en ImageNet (sin la capa superior)
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                               include_top=False,
                                               weights='imagenet')

# Congelar el modelo base para no modificar los pesos pre-entrenados
base_model.trainable = False

# Construir el modelo final
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# Compilar el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

# --- Entrenamiento ---
print("Iniciando entrenamiento...")
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS
)

# --- Guardar Modelo ---
model_save_path = 'modelo_entrenado.keras'
model.save(model_save_path)
print(f"Modelo guardado exitosamente en: {model_save_path}")

# --- Convertir a TFLite ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_save_path = 'modelo_entrenado.tflite'
with open(tflite_save_path, 'wb') as f:
  f.write(tflite_model)
print(f"Modelo TFLite guardado exitosamente en: {tflite_save_path}")

# --- Guardar Etiquetas ---
labels_path = 'labels.txt'
with open(labels_path, 'w') as f:
  for class_name in class_names:
    f.write(f"{class_name}\n")
print(f"Etiquetas guardadas exitosamente en: {labels_path}")

# --- Gráficas de Resultados ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('training_results.png')
print("Gráfica de resultados guardada en: training_results.png")




