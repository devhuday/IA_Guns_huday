import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Flatten
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

ArchiveName = "ModelsSaveIA/GunshotIaModel_v5.h5"

# Mantener las funciones auxiliares existentes (extract_spectrogram, add_noise, compute_optimal_threshold)
def add_noise(audio, noise_factor=0.01):
    """Añade ruido gaussiano al audio raw, antes de convertirlo a espectrograma"""
    noise = np.random.normal(0, noise_factor, len(audio))
    return audio + noise

def extract_spectrogram(file_path, duration=4, apply_noise=False):
    """Extrae el espectrograma del audio con opción de añadir ruido"""
    audio, sample_rate = librosa.load(file_path)
    num_samples = duration * sample_rate

    # Normalizar longitud del audio
    if len(audio) > num_samples:
        audio = audio[:num_samples]
    else:
        silence = np.zeros(num_samples - len(audio))
        audio = np.concatenate([audio, silence])
    
    # Aplicar ruido si se solicita
    if apply_noise:
        audio = add_noise(audio)

    # Generar espectrograma
    spect = librosa.stft(audio)
    spect_db = librosa.amplitude_to_db(np.abs(spect))
    return spect_db

def import_audios(audio_path, metadata, top=2499):
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
            # Aplicar data augmentation a los ejemplos de disparos
            apply_noise = class_label == 1.0 and random.random() < 0.5
            data = extract_spectrogram(file_name, apply_noise=apply_noise)
            
            features.append(data)
            labels.append(class_label)
            num_gun += (class_label == 1.0)
            num_nogun += (class_label == 0.0)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    print(f"Total samples: {len(features)}, Guns: {num_gun}, No Guns: {num_nogun}")
    return np.array(features), np.array(labels)

def create_model(input_shape):
    model = tf.keras.Sequential([
        # Primera capa convolucional
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Segunda capa convolucional
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Tercera capa convolucional
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Cuarta capa convolucional
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Flatten(),
        
        # Capas densas con dropout
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    return model

def train_with_cross_validation(X, y, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_no = 1
    histories = []
    val_scores = []
    
    for train_idx, val_idx in kfold.split(X):
        print(f'Training fold {fold_no}')
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Calcular pesos de clase para este fold
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        # Crear y compilar el modelo
        model = create_model(input_shape=(1025, 173, 1))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Entrenar el modelo
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_data=(X_val, y_val),
            class_weight=class_weight_dict,
            callbacks=[early_stopping]
        )
        
        # Evaluar el modelo
        scores = model.evaluate(X_val, y_val, verbose=0)
        print(f'Fold {fold_no} - Val Accuracy: {scores[1]:.4f}')
        
        histories.append(history.history)
        val_scores.append(scores[1])
        
        # Guardar el mejor modelo
        if fold_no == 1 or scores[1] > max(val_scores[:-1]):
            model.save(ArchiveName)
        
        fold_no += 1
    
    return histories, val_scores

# Cargar y preparar datos
metadata = pd.read_csv('Dataset/AudioTraining_v5.csv')
audio_path = '../Audio Tesis/Audio_Datasets'
features, labels = import_audios(audio_path, metadata)

# Preparar datos para el entrenamiento
X = features[..., np.newaxis]
histories, val_scores = train_with_cross_validation(X, labels)

# Visualizar resultados
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
for i, history in enumerate(histories):
    plt.plot(history['accuracy'], label=f'Fold {i+1}')
plt.title('Training Accuracy Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
for i, history in enumerate(histories):
    plt.plot(history['val_accuracy'], label=f'Fold {i+1}')
plt.title('Validation Accuracy Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Cargar el mejor modelo y hacer predicciones finales
best_model = tf.keras.models.load_model(ArchiveName)
y_pred_prob = best_model.predict(X)
y_pred_class = (y_pred_prob > 0.5).astype("int32")

# Mostrar métricas finales
print("\nMétricas finales con el mejor modelo:")
print(classification_report(labels, y_pred_class, target_names=['not_gun_shot', 'gun_shot']))

# Matriz de confusión
cm = confusion_matrix(labels, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['not_gun_shot', 'gun_shot'],
            yticklabels=['not_gun_shot', 'gun_shot'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Curva ROC
fpr, tpr, thresholds = roc_curve(labels, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()