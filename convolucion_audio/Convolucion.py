import numpy as np
import librosa
import soundfile as sf
from scipy.signal import convolve
import matplotlib.pyplot as plt

def parametros_audio(name, data, sampling):
    # Obtener Duración
    duration = librosa.get_duration(y=data, sr=sampling)

    # Imprimir los parámetros
    print(f"Audio: {name}")
    print(f"Duración: {duration} segundos")
    print(f"Frecuencia de muestreo: {sampling} Hz")
    print(f"Número de canales: {data.ndim}\n")
    
# Función para cargar un audio
def cargar_audio(ruta, sr=None):
    audio, sr = librosa.load(ruta, sr=sr)
    return audio, sr

# Función para guardar un audio
def guardar_audio(ruta, audio, sr):
    sf.write(ruta, audio, sr)

# Función para convolucionar dos audios
def convolucionar_audios(audio1, audio2):
    return convolve(audio1, audio2, mode='same')

# Función para graficar los audios
def graficar_audios(audios, labels, sr, titulo="Audios"):
    plt.figure(figsize=(12, 8))
    time = 0
    for audio in audios:
        time = len(audio) if len(audio)>time else time
    print(time)
    for i, audio in enumerate(audios):
        t = np.linspace(0, len(audio) / sr, len(audio))  # Tiempo en segundos
        plt.subplot(len(audios), 1, i + 1)
        plt.plot(t, audio)
        plt.title(labels[i])
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud")
    plt.tight_layout()
    plt.suptitle(titulo, fontsize=16, y=1.02)
    plt.show()


ruta_audio2 = "2 (10).wav"  # Primer disparo
ruta_audio1 = "Audio_Poligono_5.wav"  # Segundo disparo
ruta_salida = "resultado_convolucion.wav"  # Audio convolucionado

# Cargar los audios
audio1, sr1 = cargar_audio(ruta_audio1)
audio2, sr2 = cargar_audio(ruta_audio2)

parametros_audio(ruta_audio1, audio1, sr1)
parametros_audio(ruta_audio2, audio2, sr2)

# Asegurarse de que las tasas de muestreo coincidan
if sr1 != sr2:
    #raise ValueError("Las tasas de muestreo de los audios no coinciden.")
    print("Las tasas de muestreo de los audios no coinciden...resampling")
    audio_re_muestreado = librosa.resample(audio1, orig_sr=sr1, target_sr=sr2)
    audio1=audio_re_muestreado
    sr1 = sr2
    
# Convolucionar los audios
audio_convolucionado = convolucionar_audios(audio1, audio2)

# Normalizar el audio resultante para evitar saturación
audio_convolucionado = audio_convolucionado / np.max(90)

graficar_audios(
    [audio1, audio2, audio_convolucionado],
    ["Disparo Poligono", "Disparo datasets", "Convolución"],
    sr1,
    titulo="Audios y Convolución"
)

# Guardar el resultado
guardar_audio(ruta_salida, audio_convolucionado, sr1)

print(f"Audio convolucionado guardado en: {ruta_salida}")
