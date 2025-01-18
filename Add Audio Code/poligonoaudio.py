import os
import pandas as pd
import shutil
import sounddevice as sd
import soundfile as sf
# Rutas y configuraciones iniciales

Path = "C:/Users/HudayPlata/Documents/Unimag Tesis/Audio Tesis/"
OldFold = 'audios peligono tiros'
FoldGuns = "PoligonoTiro"
FoldNoGuns = "PoligonoNoTiro"

archivo_csv_original = 'Dataset/AudioTraining_v4.csv'
nuevo_archivo_csv = 'Dataset/AudioTraining_v5.csv'


# Obtener nombres de archivos WAV y moverlos con nombres nuevos
nombres_archivos = []
num = 0
i = 0
for archivo in os.listdir(f"{Path}{OldFold}"):
    i += 1
    print(f"{i}){archivo}")
    print(f"Reproduciendo...")
    archivo_completo = os.path.join(f"{Path}{OldFold}", archivo)
    if os.path.isfile(archivo_completo):

        audio_data, sample_rate = sf.read(archivo_completo)
        audio_data = audio_data[0:4*sample_rate]
        sd.play(audio_data, samplerate=sample_rate)

        # Generar nuevo nombre para el archivo
        disp = input("¿Es disparo? (y/n): ")
        if disp.lower() == "y":
            nueva_ruta = os.path.join(Path, FoldGuns)
            shutil.copy(archivo_completo, nueva_ruta)  # Mover archivo
        else:
            nueva_ruta = os.path.join(Path, FoldNoGuns)
            shutil.copy(archivo_completo, nueva_ruta)
    print(f"\n")

print(f"TERMINADO")
