import os

import scipy.signal
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import keras
import scipy
import noisereduce as nr 

def extract_spectogram(Audio, features):
    # Calcula el STFT del audio
    spect = librosa.stft(Audio, hop_length=1024)
    spectInDb = librosa.amplitude_to_db(spect)
    features = spectInDb
    print(f"Shape after STFT: {spectInDb.shape}")
    
    # Ajusta el tamaño del espectrograma para que coincida con las dimensiones esperadas por el modelo
    features = np.expand_dims(features, axis=-1)  # Añadir canal de profundidad
    features = np.expand_dims(features, axis=0)   # Añadir dimensión del batch
    print(f"Shape after expanding dimensions: {features.shape}")
    
    return features


# Función para capturar audio desde el micrófono
def record_audio(duration=4, sample_rate=44200):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Esperar a que termine la grabación
    sf.write("prueba1dfdsf.wav", audio, sample_rate)
    audio=audio.flatten()
    return audio, sample_rate
    
# Función para predecir si un sonido es un disparo
def predict_sound_from_mic(model):
    features = []
    audio, sample_rate = record_audio()
    features = extract_spectogram(audio, features)
    features = np.array(features)
    print(f"Shape after converting to NumPy: {features.shape}")
    prediction = (model.predict(features) > 0.3).astype("int32")
    predicted_label = 'gun_shot' if prediction == 1 else 'not_gun_shot'
    return predicted_label

# Cargar el modelo entrenado
model = keras.saving.load_model(('D:/Workspace/tesis/IACode/gunshot_detection_model_NEW.h5'))

# Loop para grabar y predecir múltiples veces3
while True:
    result = predict_sound_from_mic(model)
    print(f'Predicted Class: {result}')

    # Preguntar si se desea realizar otra predicción
    another = input("¿Deseas hacer otra prueba? (si/no): ").strip().lower()
    if another != 'si':
        print("Finalizando las pruebas.")
        break