"""
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
def record_audio(duration=4, sample_rate=44100):
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
model = keras.saving.load_model(('gunshot_detection_model_NEW.h5'))

# Loop para grabar y predecir múltiples veces3
while True:
    result = predict_sound_from_mic(model)
    print(f'Predicted Class: {result}')

    # Preguntar si se desea realizar otra predicción
    another = input("¿Deseas hacer otra prueba? (si/no): ").strip().lower()
    if another != 'si':
        print("Finalizando las pruebas.")
        break
"""
import sounddevice as sd
import numpy as np
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import soundfile as sf

# Cargar el modelo entrenado
model = load_model('GunshotIaModel_v2.h5')
model2 = load_model("IACode/GunshotIaModel_v2_1.h5")
def record_audio(duration, sample_rate=22050):
    """
    Graba un audio desde el micrófono por una duración específica.

    :param duration: Duración de la grabación en segundos.
    :param sample_rate: Frecuencia de muestreo del audio.
    :return: Audio grabado como un array de numpy.
    """
    print("Grabando...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Esperar a que termine la grabación
    print("Grabación finalizada.")
    return audio.flatten()

def play_audio(audio, sample_rate=22050):
    """
    Reproduce un audio grabado.

    :param audio: Audio como array de numpy.
    :param sample_rate: Frecuencia de muestreo del audio.
    """
    print("Reproduciendo el audio grabado...")
    sd.play(audio, samplerate=sample_rate)
    sd.wait()
    print("Reproducción finalizada.")

def process_audio(audio, sample_rate=22050, duration=4):
    """
    Ajusta la duración del audio y genera un espectrograma.

    :param audio: Audio como array de numpy.
    :param sample_rate: Frecuencia de muestreo del audio.
    :param duration: Duración deseada del audio en segundos.
    :return: Espectrograma del audio procesado.
    """
    num_samples = duration * sample_rate
    if len(audio) > num_samples:
        audio = audio[:num_samples]
    else:
        silence = np.zeros(num_samples - len(audio))
        audio = np.concatenate([audio, silence])
    
    # Crear espectrograma
    spect = librosa.stft(audio)
    spect_db = librosa.amplitude_to_db(spect, ref=np.max)
    
    # Normalizar
    spect_db = np.expand_dims(spect_db, axis=-1)  # Agregar una dimensión para el canal
    return spect_db

def predict_gunshot(spectrogram):
    """
    Realiza la predicción con el modelo entrenado.

    :param spectrogram: Espectrograma del audio.
    :return: Predicción de probabilidad.
    """
    # Ajustar el tamaño para el modelo
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Batch dimension
    prob = model.predict(spectrogram)[0][0]
    prob2 = model2.predict(spectrogram)[0][0]
    return prob, prob2

# Grabar audio desde el micrófono
duration = 4  # Duración en segundos
sample_rate = 22050
audio = record_audio(duration, sample_rate)

# Guardar el audio grabado para inspección si es necesario
sf.write("audio_grabado.wav", audio, sample_rate)

# Reproducir el audio grabado
play_audio(audio, sample_rate)

# Procesar el audio
spectrogram = process_audio(audio, sample_rate, duration)

# Mostrar el espectrograma para verificar visualmente
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram[:, :, 0], sr=sample_rate, x_axis='time', y_axis='hz', cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
plt.title("Espectrograma del Audio Grabado")
plt.show()

# Realizar la predicción
prob, prob2 = predict_gunshot(spectrogram)
print(f"Probabil  idad de ser un disparo modelo 1: {prob:.3f}")
print(f"Probabilidad de ser un disparo modelo 2: {prob2:.3f}")

# Interpretar la predicción
if prob > 0.5:
    print("¡Se detectó un disparo!")
else:
    print("No se detectó un disparo.")
