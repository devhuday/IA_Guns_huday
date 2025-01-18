import socket
import numpy as np
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd

# Cargar el modelo entrenado
model = load_model('gunshot_detection_model_110125.h5')
model2 = load_model("GunshotIaModel_v1.0.h5")
model3 = load_model("GunshotIaModel_v1.5.h5")

def record_audio_from_wifi(esp32_ip, port, duration, sample_rate=22050, chunk_size=256):
    """
    Lee audio desde una ESP32 a través de WiFi por una duración específica.

    :param esp32_ip: Dirección IP de la ESP32.
    :param port: Puerto en el que escucha el servidor en la ESP32.
    :param duration: Duración de la captura en segundos.
    :param sample_rate: Frecuencia de muestreo del audio.
    :param chunk_size: Tamaño de cada segmento de datos recibido.
    :return: Audio grabado como un array de numpy.
    """
    print(f"Conectando al servidor en {esp32_ip}:{port}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((esp32_ip, port))
    print("Conexión establecida.")

    num_samples = int(duration * sample_rate)
    audio = []

    print("Grabando audio desde WiFi...")
    try:
        while len(audio) < num_samples:
            # Recibir datos en bruto desde la ESP32
            raw_data = client_socket.recv(chunk_size * 2)  # 2 bytes por muestra (16 bits)
            if raw_data:
                # Convertir los datos recibidos a enteros de 16 bits
                chunk = np.frombuffer(raw_data, dtype=np.int16)
                audio.extend(chunk)
            else:
                break
    finally:
        client_socket.close()
        print("Conexión cerrada.")

    # Normalizar el audio a rango [-1, 1]
    audio = np.array(audio[:num_samples], dtype=np.float32)
    audio /= np.max(np.abs(audio))  # Normalización
    print("Grabación finalizada.")
    return audio

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
    prob3 = model3.predict(spectrogram)[0][0]
    return prob, prob2, prob3

# Configuración para grabar audio desde la ESP32 vía WiFi
esp32_ip = '192.168.1.102'  # Cambiar por la IP de tu ESP32
port = 8082
duration = 4  # Duración en segundos
sample_rate = 22050
chunk_size = 256

# Grabar audio desde WiFi
audio = record_audio_from_wifi(esp32_ip, port, duration, sample_rate, chunk_size)

# Guardar el audio grabado para inspección si es necesario
sf.write("audio_grabado_wifi.wav", audio, sample_rate)

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
prob, prob2, prob3 = predict_gunshot(spectrogram)
print(f"Probabil  idad de ser un disparo modelo 1: {prob:.3f}")
print(f"Probabilidad de ser un disparo modelo 2: {prob2:.3f}")
print(f"Probabilidad de ser un disparo modelo 3: {prob3:.3f}")
# Interpretar la predicción
if prob > 0.5:
    print("¡Se detectó un disparo!")
else:
    print("No se detectó un disparo.")