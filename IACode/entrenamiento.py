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

ArchiveName = "GunshotIaModel_v1.0.h5"

#%%
# Funciones auxiliares para preprocesamiento y augmentación de datos
def extract_spectrogram(file_path, duration=4):
    audio, sample_rate = librosa.load(file_path)
    num_samples = duration * sample_rate

    if len(audio) > num_samples:
        audio = audio[:num_samples]
    else:
        silence = np.zeros(num_samples - len(audio))
        audio = np.concatenate([audio, silence])

    spect = librosa.stft(audio)
    spect_db = librosa.amplitude_to_db(spect)
    return spect_db

def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def import_audios(audio_path, metadata, top=1999):
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
    return np.array(features), np.array(labels)

def compute_optimal_threshold(y_test, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]


#%%
# Cargar metadatos y datos
metadata = pd.read_csv('Dataset/AudioTraining_v5.csv')
audio_path = 'C:/Users/HudayPlata/Documents/Unimag Tesis/Audio Tesis/Audio_Datasets'
features, labels = import_audios(audio_path, metadata)


#%%
# Convertir las listas en arrays de numpy
X = features[..., np.newaxis]  # Agregar dimensión de canal

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Calcular pesos de clase
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}


#%%
# Crear el modelo CRNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(1025, 173, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())


#%%
# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=16, class_weight=class_weight_dict)


#%%
model.save(ArchiveName)
# Evaluar el modelo
score = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy: {score[1]}")

# Predicciones y probabilidades
y_pred_prob = model.predict(X_test)
y_pred_class = (y_pred_prob > 0.5).astype("int32")



#%% 
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
fpr, tpr, thresholds = roc_curve(y_test, 
                                 y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Umbral óptimo
optimal_threshold = compute_optimal_threshold(y_test, y_pred_prob)
print(f"Optimal Threshold: {optimal_threshold}")

# %%
