import librosa
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
import h5py

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_and_save_data(audio_path, metadata, output_file, duration=4, n_fft=2048, hop_length=512):
    """Preprocesa todos los archivos de audio y los guarda en un archivo H5"""
    with h5py.File(output_file, 'w') as hf:
        # Crear datasets
        num_samples = len(metadata)
        hf.create_dataset('spectrograms', (num_samples, 1025, 173, 1), dtype='float32')
        hf.create_dataset('labels', (num_samples,), dtype='float32')
        
        for idx, (_, row) in enumerate(metadata.iterrows()):
            if idx % 100 == 0:
                print(f'Procesando archivo {idx}/{num_samples}')
                
            file_name = os.path.join(audio_path, f"fold{row['fold']}", row['slice_file_name'])
            try:
                spect = extract_spectrogram(file_name, duration, n_fft, hop_length)
                hf['spectrograms'][idx] = spect[..., np.newaxis]
                hf['labels'][idx] = 1.0 if row['classID'] == 6 else 0.0
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

def data_generator_from_cache(h5_file, batch_size=32, shuffle=True):
    """Generator que lee datos preprocesados del archivo H5"""
    with h5py.File(h5_file, 'r') as hf:
        indices = np.arange(len(hf['labels']))
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                X = hf['spectrograms'][batch_indices]
                y = hf['labels'][batch_indices]
                yield X, y

def extract_spectrogram(file_path, duration=4, n_fft=2048, hop_length=512):
    audio, sample_rate = librosa.load(file_path, sr=None)
    num_samples = duration * sample_rate
    if len(audio) > num_samples:
        audio = audio[:num_samples]
    else:
        silence = np.zeros(num_samples - len(audio))
        audio = np.concatenate([audio, silence])
    
    # Usar librosa.stft con dtype float32 para reducir uso de memoria
    spect = librosa.stft(audio.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
    spect_db = librosa.amplitude_to_db(np.abs(spect))
    spect_db = spect_db[:1025, :173]
    if spect_db.shape[1] < 173:
        spect_db = np.pad(spect_db, ((0, 0), (0, 173 - spect_db.shape[1])), mode='constant')
    return spect_db
 
def data_generator(audio_path, metadata, batch_size=32, duration=4, n_fft=2048, hop_length=512, class_weights=None):
    """Modified generator that incorporates class weights into the batch"""
    samples = metadata.copy()  # Create a copy to avoid modifying original
    while True:
        # Shuffle the samples at the start of each epoch
        samples = samples.sample(frac=1).reset_index(drop=True)
        features, labels, sample_weights = [], [], []
        
        for _, row in samples.iterrows():
            file_name = os.path.join(audio_path, f"fold{row['fold']}", row['slice_file_name'])
            class_label = 1.0 if row['classID'] == 6 else 0.0
            
            try:
                spect = extract_spectrogram(file_name, duration, n_fft, hop_length)
                features.append(spect[..., np.newaxis])
                labels.append(class_label)
                
                # Add sample weight based on class
                if class_weights is not None:
                    weight = class_weights[1] if class_label == 1.0 else class_weights[0]
                    sample_weights.append(weight)
                
                if len(features) == batch_size:
                    if class_weights is not None:
                        yield (np.array(features, dtype=np.float32), 
                              np.array(labels, dtype=np.float32), 
                              np.array(sample_weights, dtype=np.float32))
                    else:
                        yield (np.array(features, dtype=np.float32), 
                              np.array(labels, dtype=np.float32))
                    features, labels, sample_weights = [], [], []
                    
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                continue


# Configurar memoria de GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Cargar metadatos
metadata = pd.read_csv('Dataset/AudioTraining_v5.csv')
audio_path = 'C:/Users/HudayPlata/Documents/Unimag Tesis/Audio Tesis/Audio_Datasets'

# Dividir datos
train_metadata, test_metadata = train_test_split(metadata, test_size=0.3, random_state=42)

# Preprocesar y guardar datos
preprocess_and_save_data(audio_path, train_metadata, 'train_data.h5')
preprocess_and_save_data(audio_path, test_metadata, 'test_data.h5')

# Crear modelo con batch normalization y dropout para mejor rendimiento
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(1025, 173, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Reshape((-1, 64)),
    
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar con optimizador más eficiente
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Crear generadores desde caché
train_gen = data_generator_from_cache('train_data.h5', batch_size=32)
test_gen = data_generator_from_cache('test_data.h5', batch_size=32)

# Entrenar modelo
steps_per_epoch = len(train_metadata) // 32
validation_steps = len(test_metadata) // 32

history = model.fit(
    train_gen,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_gen,
    validation_steps=validation_steps,
    workers=4,
    use_multiprocessing=True
)


# Evaluation
test_steps = len(test_metadata) // 32
test_data, test_labels = next(iter(data_generator(audio_path, test_metadata, batch_size=len(test_metadata))))
predictions = model.predict(test_data)

# Convert probabilities to labels
predicted_labels = (predictions > 0.5).astype(int)

# Generate confusion matrix and classification report
conf_matrix = confusion_matrix(test_labels[:len(predicted_labels)], predicted_labels)
report = classification_report(test_labels[:len(predicted_labels)], predicted_labels, 
                             target_names=['No Gunshot', 'Gunshot'])

# ROC curve
fpr, tpr, _ = roc_curve(test_labels[:len(predicted_labels)], predictions)
roc_auc = auc(fpr, tpr)

# Show results
print(report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['No Gunshot', 'Gunshot'], 
            yticklabels=['No Gunshot', 'Gunshot'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()