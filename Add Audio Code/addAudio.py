import os
import pandas as pd
import shutil

# Rutas y configuraciones iniciales

Path = "C:/Users/HudayPlata/Documents/Unimag Tesis/Audio Tesis/"
OldFold = 'PoligonoTiro'
classAudio = 'gun_shot'
classID = 6
numberfold = '13'
nuevo_directorio = f"{Path}Audio_Datasets/fold{numberfold}"
archivo_csv_original = 'Dataset/AudioTraining_v6.csv'
nuevo_archivo_csv = 'Dataset/AudioTraining_v7.csv'

# Crear la carpeta si no existe
os.makedirs(nuevo_directorio, exist_ok=True)

# Obtener nombres de archivos WAV y moverlos con nombres nuevos
nombres_archivos = []
num = 0
for archivo in os.listdir(f"{Path}{OldFold}"):
    num+=1
    if num > 99:
        break
    archivo_completo = os.path.join(f"{Path}{OldFold}", archivo)
    if os.path.isfile(archivo_completo):
        # Generar nuevo nombre para el archivo
        nuevo_nombre = f"{OldFold}{num}{archivo}"
        nueva_ruta = os.path.join(nuevo_directorio, nuevo_nombre)
        shutil.copy(archivo_completo, nueva_ruta)  # Mover archivo
        nombres_archivos.append(nuevo_nombre)  # Guardar nueva ruta del archivo

# Leer archivo CSV original
Sound_data = pd.read_csv(archivo_csv_original)

# Crear un DataFrame con los nuevos datos
nuevos_datos = pd.DataFrame({
    'slice_file_name': nombres_archivos,
    'fsID': 0,
    'start': 0,
    'end': 0,
    'salience': 0,
    'fold': numberfold,
    'classID': classID,
    'class': classAudio
})

# Concatenar los datos originales con los nuevos datos
archivo_final = pd.concat([Sound_data, nuevos_datos], ignore_index=True)

# Guardar el DataFrame combinado en un nuevo archivo CSV
archivo_final.to_csv(nuevo_archivo_csv, index=False)

print(f"Se ha creado el archivo combinado en: {nuevo_archivo_csv}")
print(f"Los archivos procesados se han movido a: {nuevo_directorio}")
