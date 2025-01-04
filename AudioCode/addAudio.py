import os
import pandas as pd
Path = "C:/Users/HudayPlata/Documents/Unimag Tesis/Audio Tesis/"

AudioFold = 'BackSound'
archivo_csv_original = 'AudioTraining_v2.csv'
nuevo_archivo_csv = 'AudioTraining_v3.csv'

nombres_archivos = [AudioFold+archivo[24:] for archivo in os.listdir(f"{Path}{AudioFold}") if os.path.isfile(os.path.join(f"{Path}{AudioFold}",archivo))]

Sound_data = pd.read_csv(archivo_csv_original)

nuevos_datos = pd.DataFrame({
    'slice_file_name': nombres_archivos,
    'fsID': 0,
    'start': 0,
    'end': 0,
    'salience': 0,
    'fold': 12,
    'classID': 6,
    'class': 'gun_shot'
})

# Concatenar los datos originales con los nuevos datos
archivo_final = pd.concat([Sound_data, nuevos_datos], ignore_index=True)

# Guardar el DataFrame combinado en un nuevo archivo CSV en el escritorio

archivo_final.to_csv(nuevo_archivo_csv, index=False)

print(f"Se ha creado el archivo combinado en: {nuevo_archivo_csv}")