import serial
import pyaudio
import time
from scipy.io.wavfile import write
import numpy as np
import threading

# Configuración del puerto serial
puerto = 'COM3'  # Cambia según tu sistema ('/dev/ttyUSB0' en Linux)
baudrate = 4e6  # Velocidad en bps
sample = []
nameFile = "casalobato_5m_GLOBO"
# Iniciar la conexión serial
try:
    esp32 = serial.Serial(port=puerto, baudrate=baudrate, timeout=0.1)
except serial.SerialException as e:
    print(f"Error al abrir el puerto serial: {e}")
    exit()

# Configurar PyAudio
audio = pyaudio.PyAudio()
sample_rate = 11000  # Frecuencia de muestreo (debe coincidir)
chunk = 512  # Tamaño del buffer de 
duration = 11
audioTotal = (sample_rate // chunk) * duration
samples_chunk = 0


stream = audio.open(format=pyaudio.paUInt8,  # Formato de 8 bits sin signo
                    channels=1,  # Audio mono
                    rate=sample_rate,
                    output=True)

print("Reproduciendo audio... Presiona Ctrl+C para detener.")
i = 0
try:
    buffer = bytearray()  # Buffer temporal para almacenar datos

    while True:
        # Leer datos disponibles del puerto serial
        data = esp32.read(esp32.in_waiting or 1)  # Leer los datos disponibles
        buffer.extend(data)  # Añadir datos al buffer

        # Enviar datos al stream si tenemos un chunk completo
        if len(buffer) >= chunk:
            samples_chunk += 1
            sample.extend(buffer[:chunk])
            #print(len(sample))
            stream.write(bytes(buffer[:chunk]))  # Reproducir el chunk
            buffer = buffer[chunk:]  # Eliminar los datos ya reproducidos
        
        if samples_chunk >= audioTotal:
            write(f"{nameFile}{i}.wav",sample_rate,np.array(sample).astype(np.uint8))
            i += 1
            samples_chunk = 0
            sample = []
            print("grabado")
               
except KeyboardInterrupt:
    print("\nReproducción detenida por el usuario.")
finally:
    # Cerrar el stream y la conexión serial
    stream.stop_stream()
    stream.close()
    audio.terminate()
    esp32.close()
