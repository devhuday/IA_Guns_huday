import librosa
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Función para extraer espectrogramas
def extract_spectrogram(file_path, duration=4, n_fft=2048, hop_length=512):
    audio, sample_rate = librosa.load(file_path, sr=None)
    num_samples = duration * sample_rate
    if len(audio) > num_samples:
        audio = audio[:num_samples]
    else:
        silence = np.zeros(num_samples - len(audio))
        audio = np.concatenate([audio, silence])
    spect = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spect_db = librosa.amplitude_to_db(np.abs(spect))
    spect_db = spect_db[:1025, :173]  # Ajustar dimensiones
    if spect_db.shape[1] < 173:
        spect_db = np.pad(spect_db, ((0, 0), (0, 173 - spect_db.shape[1])), mode='constant')
    return spect_db

# Generador para cargar datos por lotes
def data_generator(audio_path, metadata, batch_size=32, duration=4, n_fft=2048, hop_length=512):
    samples = metadata.iterrows()
    while True:
        features, labels = [], []
        for _, row in samples:
            file_name = os.path.join(audio_path, f"fold{row['fold']}", row['slice_file_name'])
            class_label = 1.0 if row['classID'] == 6 else 0.0
            try:
                spect = extract_spectrogram(file_name, duration, n_fft, hop_length)
                features.append(spect[..., np.newaxis])  # Añadir canal
                labels.append(class_label)
                if len(features) == batch_size:
                    yield np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)
                    features, labels = [], []
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

# Cargar metadatos
metadata = pd.read_csv('Dataset/AudioTraining_v5.csv')
audio_path = 'C:/Users/HudayPlata/Documents/Unimag Tesis/Audio Tesis/Audio_Datasets'

# Limitar muestras por clase
limit = 2000
guns = metadata[metadata['classID'] == 6]
no_guns = metadata[metadata['classID'] != 6]

# Ajustar límite según disponibilidad
guns_sample = guns.sample(n=min(limit, len(guns)), random_state=42)
no_guns_sample = no_guns.sample(n=min(limit, len(no_guns)), random_state=42)

# Combinar y mezclar datos balanceados
balanced_metadata = pd.concat([guns_sample, no_guns_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

# Dividir datos en entrenamiento y prueba
train_metadata, test_metadata = train_test_split(balanced_metadata, test_size=0.3, random_state=42)

# Crear generadores
train_gen = data_generator(audio_path, train_metadata)
test_gen = data_generator(audio_path, test_metadata)

# Crear el modelo CRNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(1025, 173, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Reshape((-1, 64)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

# Entrenar el modelo
steps_per_epoch = len(train_metadata) // 32
validation_steps = len(test_metadata) // 32

model.fit(train_gen, epochs=10, steps_per_epoch=steps_per_epoch, 
          validation_data=test_gen, validation_steps=validation_steps)

# Guardar el modelo
model.save("GunshotIaModel_v3.h5")

# Evaluar el modelo
test_steps = len(test_metadata) // 32
test_data, test_labels = next(test_gen)
predictions = model.predict(test_data)

# Generar métricas de evaluación
predicted_labels = (predictions > 0.5).astype(int)

conf_matrix = confusion_matrix(test_labels[:len(predicted_labels)], predicted_labels)
report = classification_report(test_labels[:len(predicted_labels)], predicted_labels, target_names=['No Gunshot', 'Gunshot'])

fpr, tpr, _ = roc_curve(test_labels[:len(predicted_labels)], predictions)
roc_auc = auc(fpr, tpr)

print(report)

# Visualizar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Gunshot', 'Gunshot'], yticklabels=['No Gunshot', 'Gunshot'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Visualizar curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()
