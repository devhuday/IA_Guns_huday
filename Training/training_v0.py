#%%
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

#%%
ArchiveName = "GunshotIaModel_v2.h5"

# Funciones auxiliares para preprocesamiento y augmentación de datos
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
    # Normalizamos la forma del espectrograma
    spect_db = spect_db[:1025, :]
    if spect_db.shape[1] < 173:
        pad_width = 173 - spect_db.shape[1]
        spect_db = np.pad(spect_db, ((0, 0), (0, pad_width)), mode='constant')
    elif spect_db.shape[1] > 173:
        spect_db = spect_db[:, :173]
    return spect_db

def add_noise(audio, noise_factor=0.01):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def import_audios(audio_path, metadata, top=2199):
    metadata = metadata.sample(frac=1, random_state=random.randint(1, 10000)).reset_index(drop=True)
    features, labels = [], []
    num_gun, num_nogun = 0, 0

    for index, row in metadata.iterrows():
        file_name = os.path.join(audio_path, f"fold{row['fold']}", row['slice_file_name'])
        class_id = row['classID']
        class_label = 1.0 if class_id == 6 else 0.0

        if num_gun > top and class_label == 1.0:
            continue
        if num_nogun > top and class_label == 0.0:
            continue

        try:
            data = extract_spectrogram(file_name)
            features.append(data)
            labels.append(class_label)
            num_gun += (class_label == 1.0)
            num_nogun += (class_label == 0.0)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    print(f"Total samples: {len(features)}, Guns: {num_gun}, No Guns: {num_nogun}")
    features = np.array(features, dtype=object)
    labels = np.array(labels)
    return features, labels

def compute_optimal_threshold(y_test, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

def data_generator(audio_path, metadata, batch_size, duration=4, max_samples=2000):
    total_samples = 0  # Contador global para los audios procesados

    while total_samples < max_samples:
        batch_metadata = metadata.sample(n=batch_size)
        features, labels = [], []
        
        for _, row in batch_metadata.iterrows():
            if total_samples >= max_samples:
                break
            
            file_name = os.path.join(audio_path, f"fold{row['fold']}", row['slice_file_name'])
            class_id = row['classID']
            class_label = 1.0 if class_id == 6 else 0.0
            
            try:
                data = extract_spectrogram(file_name, duration=duration)
                features.append(data)
                labels.append(class_label)
                total_samples += 1  # Incrementa el contador global
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
        
        if features:  # Solo retorna un batch si hay datos válidos
            yield np.array(features)[..., np.newaxis], np.array(labels)
        else:
            break  # Detén el generador si no hay datos válidos en este batch

    print(f"Total samples processed: {total_samples}")


#%%
# Cargar metadatos y datos
metadata = pd.read_csv('Dataset/AudioTraining_v5.csv')
audio_path = 'C:/Users/HudayPlata/Documents/Unimag Tesis/Audio Tesis/Audio_Datasets'
features, labels = import_audios(audio_path, metadata)
features = np.array(features, dtype=np.float32)


#%%
# Convertir las listas en arrays de numpy
X = np.array(features)[..., np.newaxis]  # Agregar dimensión de canal

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)


#%%
# Calcular pesos de clase
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("creando modelo...")
# Crear el modelo CRNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(1025, 173, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),
    tf.keras.layers.Reshape((509, 41, 64, 1)), # Agregar una dimensión de canal para ConvLSTM2D
    tf.keras.layers.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True),
    tf.keras.layers.Flatten(), # Aplanar la salida de ConvLSTM2D antes de la capa densa
    tf.keras.layers.Dense(1, activation='sigmoid')
])
print("compilando...")
#%%
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

batch_size = 32
max_samples = 2000
steps_per_epoch = max_samples // batch_size

train_gen = data_generator(audio_path, metadata, batch_size, max_samples=max_samples)
model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=5, class_weight=class_weight_dict)


# Entrenar el modelo
model.fit(X_train, y_train, epochs=5, batch_size=10, class_weight=class_weight_dict)

model.save(ArchiveName)


#%%
# Evaluar el modelo
score = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy: {score[1]}")

# Predicciones y probabilidades
y_pred_prob = model.predict(X_test)
y_pred_class = (y_pred_prob > 0.5).astype("int32")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_class)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['not_gun_shot', 'gun_shot'], yticklabels=['not_gun_shot', 'gun_shot'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Reporte de clasificación
print(classification_report(y_test, y_pred_class, target_names=['not_gun_shot', 'gun_shot']))

# Curva ROC y cálculo de AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Umbral óptimo
optimal_threshold = compute_optimal_threshold(y_test, y_pred_prob)
print(f"Optimal Threshold: {optimal_threshold}")