import serial
import pyaudio
import time
from scipy.io.wavfile import write
import numpy as np
import threading
import socket

# Configuración del socket UDP
IP = "0.0.0.0"  # Escuchar en todas las interfaces disponibles
PUERTO = 12345   # Puerto por donde recibiremos los datos
BUFFER_SIZE = 512  # Tamaño del paquete a recibir

# Crear el socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((IP, PUERTO))

# Configuración del puerto serial
puerto = 'COM3'  # Cambia según tu sistema ('/dev/ttyUSB0' en Linux)
baudrate = 4e6  # Velocidad en bps
sample = []
nameFile = "casalobato_5m_GLOBO"

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
        data, address = sock.recvfrom(BUFFER_SIZE)  
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
