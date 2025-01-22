import socket
import numpy as np
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
import threading

# Cargar el modelo entrenado
model = load_model('GunshotIaModel_v2_1.h5')

def record_audio_from_wifi(esp32_ip, port, duration, sample_rate=22050, chunk_size=256):
    """
    Lee audio desde una ESP32 a través de WiFi por una duración específica.
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
            raw_data = client_socket.recv(chunk_size * 2)
            if raw_data:
                chunk = np.frombuffer(raw_data, dtype=np.int16)
                audio.extend(chunk)
            else:
                break
    finally:
        client_socket.close()
        print("Conexión cerrada.")

    audio = np.array(audio[:num_samples], dtype=np.float32)
    audio /= np.max(np.abs(audio))
    print("Grabación finalizada.")
    return audio

def play_audio(audio, sample_rate=22050):
    """
    Reproduce un audio grabado.
    """
    print("Reproduciendo el audio grabado...")
    sd.play(audio, samplerate=sample_rate)
    sd.wait()
    print("Reproducción finalizada.")

def plot_spectrogram(audio, sample_rate, title, device_name):
    """
    Genera y guarda el espectrograma del audio.
    
    :param audio: Audio como array de numpy
    :param sample_rate: Frecuencia de muestreo
    :param title: Título del gráfico
    :param device_name: Nombre del dispositivo ESP32
    """
    plt.figure(figsize=(10, 4))
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    librosa.display.specshow(S_db, sr=sample_rate, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{title} - {device_name}')
    
    # Guardar el espectrograma
    plt.savefig(f'spectrogram_{device_name}.png')
    plt.close()

def process_audio(audio, sample_rate=22050, duration=4):
    """
    Ajusta la duración del audio y genera un espectrograma.
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
    spect_db = np.expand_dims(spect_db, axis=-1)
    return spect_db

def predict_gunshot(spectrogram):
    """
    Realiza la predicción con el modelo entrenado.
    """
    spectrogram = np.expand_dims(spectrogram, axis=0)
    prob = model.predict(spectrogram)[0][0]
    return prob

def procesing(esp32_ip, port, duration, sample_rate, chunk_size, nombre):
    print(f"N: {nombre}")
    # Grabar audio desde WiFi
    audio = record_audio_from_wifi(esp32_ip, port, duration, sample_rate, chunk_size)
    
    # Guardar el audio grabado
    sf.write(f"audio_grabado_{nombre}.wav", audio, sample_rate)
    
    # Generar y guardar el espectrograma
    plot_spectrogram(audio, sample_rate, "Espectrograma del Audio", nombre)
    
    # Reproducir el audio grabado
    play_audio(audio, sample_rate)
    
    # Procesar el audio
    spectrogram = process_audio(audio, sample_rate, duration)
    
    # Realizar la predicción
    prob = predict_gunshot(spectrogram)
    print(f"Probabilidad de ser un disparo: {prob:.2f}")
    if prob > 0.5:
        print(f"¡Se detectó un disparo! {nombre}")
    else:
        print(f"No se detectó un disparo {nombre}")

# Configuración principal
duration = 4  # Duración en segundos
sample_rate = 22050
chunk_size = 256

# Configuración de dispositivos ESP32
esp32_divice_1 = ['192.168.1.100', 80, "ESP32_1"]
esp32_divice_2 = ['192.168.1.102', 8082, "ESP32_2"]
esp32_divice_3 = ['192.168.1.103', 8083, "ESP32_3"]

# Crear y ejecutar hilos
hilo_esp_1 = threading.Thread(target=procesing, args=(esp32_divice_1[0], esp32_divice_1[1], duration, sample_rate, chunk_size, esp32_divice_1[2]))
hilo_esp_2 = threading.Thread(target=procesing, args=(esp32_divice_2[0], esp32_divice_2[1], duration, sample_rate, chunk_size, esp32_divice_2[2]))
hilo_esp_3 = threading.Thread(target=procesing, args=(esp32_divice_3[0], esp32_divice_3[1], duration, sample_rate, chunk_size, esp32_divice_3[2]))

hilo_esp_1.start()
hilo_esp_2.start()
hilo_esp_3.start()

hilo_esp_1.join()
hilo_esp_2.join()
hilo_esp_3.join()