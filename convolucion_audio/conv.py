import sounddevice as sd
import soundfile as sf
import numpy as np
import scipy.signal as signal
import librosa
import matplotlib.pyplot as plt
import plotly.offline as pyo
import plotly.express as px
import os
from scipy.fft import fft, ifft

# Variables globales para almacenar las rutas de los archivos
archivo1 = None
archivo2 = None
archivo_conv = None
sr_conv = None

# Función para seleccionar un archivo y mostrar la ruta en el label correspondiente
def seleccionar_archivo(label_text, path):
    global archivo1, archivo2
    archivo = path
    archivo_name = archivo.split('/')[-1]
    archivo_name = f'{archivo_name[:12]}...' if len(archivo_name) > 18 else archivo_name
    print(f"archivo: {archivo_name}")
    if label_text == "Archivo 1":
        archivo1 = archivo_name
    elif label_text == "Archivo 2":
        archivo2 = archivo_name
        
# Función para normalizar el audio cargado
def normalize(archivo):
    y, sr = sf.read(archivo)
    max = np.max(np.absolute(y))
    y_norm = y / max
    return y_norm, sr

# Función para reproducir el archivo seleccionado y mostrar la onda de audio
def reproducir_archivo(text = None, path = None):
    archivo = path
    x, sr = normalize(archivo)
    print(f'archivo: {path}')
    print(f'numero de canales: {sf.SoundFile(archivo).channels}')
    x = (x[:,0]+x[:,1])/2 if sf.SoundFile(archivo).channels > 1 else x
    print(f'numero de muestras: {x.shape[0]}')
    print(f'sample rate: {sr}\n')
    
    #sd.play(x, sr)
    #sd.wait()
    print("Reproducción finalizada.")
    return x, sr

def plot(name, signal, samplerate):
    print(f'se plotea {name}')
    signalarray, freq = signal, samplerate
    samples = len(signalarray)
    duration = samples / freq
    time = np.linspace(0, duration, samples)
    fig = px.line(x = time,
                y = signalarray,
                title = f'Amplitud x Tiempo {name}',
                labels={'x':'Tiempo (min)', 'y':'Potencia (dB)'}
                ) #Espectro de potencia
    pyo.plot(fig, filename=f'plot{name}_1y.html')

# Función para realizar la convolución de dos archivos de audio y reproducir el resultado
def convolucionar(Nx, Ny, sr1):
    
    # Realizar la convolución
    convolution = signal.convolve(Nx, Ny, mode='full')
    convolution = np.convolve(Nx, Ny, 'full')
    # Reproducir el resultado
    max = np.max(np.absolute(convolution))
    convolution = convolution / max
    #
    # convolution = convolution[3*sr1:]
    sd.play(convolution, sr1)
    sd.wait()
    print("Reproducción finalizada conv.")
    
    sr_conv = sr1
    archivo_conv = convolution
    return archivo_conv, sr_conv

def Guardar_audio(path):
    if sr_conv:
        archivoPath = path
        print(archivoPath)
        if archivoPath:
            path = archivoPath if archivoPath[-4:] == '.wav' else f'{archivoPath}.wav'
            sf.write(path,archivo_conv,sr_conv)

def deconvolucionar (x_gun, x_noise, sr1):
    
    # Ajustar la longitud de ambas señales para que coincidan
    len_disparo = len(x_gun)
    len_ruido = len(x_noise)

    # Si las longitudes son diferentes, rellenar con ceros la señal más corta
    if len_disparo < len_ruido:
        # Rellenar con ceros al final de 'disparo'
        x_gun = np.pad(x_gun, (0, len_ruido - len_disparo), mode='constant')
    elif len_disparo > len_ruido:
        # Rellenar con ceros al final de 'ruido_recinto'
        ruido_recinto = np.pad(ruido_recinto, (0, len_disparo - len_ruido), mode='constant')  
    
    disparo_fft = fft(x_gun)
    ruido_fft = fft(x_noise)
    
    H = ruido_fft / (disparo_fft)
    respuesta_al_impulso = ifft(H)
    respuesta_al_impulso = np.real(respuesta_al_impulso)
    respuesta_al_impulso /= np.max(np.abs(respuesta_al_impulso))
    
    sf.write('respuesta_impulso_recinto.wav', respuesta_al_impulso, sr1)
    sd.play(respuesta_al_impulso, sr1)
    sd.wait()
    
    print("Reproducción finalizada deconv.")
    return respuesta_al_impulso, sr1
    
def save_audio(data, samplerate, pahtname):
    sf.write(pahtname,data,samplerate)

pyo.init_notebook_mode(connected=True)

os.system('cls')

#generamos las rutas para la lectura
path_general = 'D:/Workspace/tesis'
archivo_noise = "Audio_Poligono_5.wav"
archivo_gun = '2 (10).wav'
archivo_conv = 'convolucion 1'

#pathp2 = f'D:/Workspace/tesis/addAudio/testing convolution/convolucion 1y.wav'

#pathp2 = f'{path}field-open-20.wav'
pathp1 = os.path.join(path_general, archivo_gun)
pathp2 = os.path.join(path_general,'audioMetadata','Audio',archivo_noise)
#pathp2 = os.path.join(path_general,'convolucion 1.wav')
pathp2 = 'respuesta_impulso_recinto.wav'
pathp2 = os.path.join(path_general,archivo_noise)
#disparo
x1, sr1 = reproducir_archivo(path=pathp1)
plot1 = plot(archivo_gun, x1, sr1)

#ruido
x2, sr2 = reproducir_archivo(path=pathp2)
#plot2 = plot(archivo_noise, x2, sr2)
step = len(x1)/sr1
print(step)
print(len(x2))
x2 = x2[:int(step*sr2)]
print(len(x2))
print(len(x1))
y1, sry1 = convolucionar(x1*10, x2, sr1)
plot3 = plot(f'{archivo_conv}', y1, sry1)

save_audio(y1, sry1, f'{path_general}/{archivo_conv}.wav')

#y2, sry2 = deconvolucionar(x1, x2, sr1)
#plot4 = plot("respuesta al impulso", y2, sry2)