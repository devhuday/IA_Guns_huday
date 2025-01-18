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
import tensorflow.keras.backend as K

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

def import_audios(audio_path, metadata, top=199):
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

def test_data_generator(audio_path, metadata, batch_size):
    for start in range(0, len(metadata), batch_size):
        end = min(start + batch_size, len(metadata))
        batch_metadata = metadata[start:end]
        features, labels = [], []
        
        for _, row in batch_metadata.iterrows():
            file_name = os.path.join(audio_path, f"fold{row['fold']}", row['slice_file_name'])
            class_label = 1.0 if row['classID'] == 6 else 0.0
            try:
                data = extract_spectrogram(file_name)
                features.append(data)
                labels.append(class_label)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

        features = np.array(features, dtype=np.float32)[..., np.newaxis]
        labels = np.array(labels)
        yield features, labels

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.convert_to_tensor([class_weight_dict[0], class_weight_dict[1]])
    y_true = tf.cast(y_true, tf.int32)
    class_weights = tf.gather(weights, y_true)
    bce = K.binary_crossentropy(y_true, y_pred)
    return bce * class_weights

ArchiveName = "GunshotIaModel_v4.h5"

# Cargar metadatos
metadata = pd.read_csv('Dataset/AudioTraining_v5.csv')
audio_path = 'C:/Users/HudayPlata/Documents/Unimag Tesis/Audio Tesis/Audio_Datasets'

# Extraer las etiquetas de la columna correspondiente
labels = metadata['classID'].values  # Reemplaza 'Label' con el nombre real de la columna de etiquetas en tu CSV

# Calcular pesos de clase
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

print("Class Weights:", class_weight_dict)

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
model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=['accuracy'])
print(model.summary())

batch_size = 32
max_samples = 2000

train_gen = data_generator(audio_path, metadata, batch_size, max_samples=max_samples)
steps_per_epoch = max_samples // batch_size

# Entrenamiento del modelo
model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=5)

model.save(ArchiveName)
