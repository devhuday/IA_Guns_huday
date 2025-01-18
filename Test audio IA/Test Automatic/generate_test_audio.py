import os
import csv
import sounddevice as sd
from scipy.io.wavfile import write
import time

def main():
    # Crear carpeta 'testing' si no existe
    folder_name = "testing"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Nombre del archivo CSV
    csv_file = os.path.join(folder_name, "labels.csv")

    # Crear archivo CSV con encabezados si no existe
    if not os.path.isfile(csv_file):
        with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "is_gunshot"])

    print("\n--- Audio Recorder ---")
    print("Graba audios de 4 segundos y etiqueta si es un disparo o no.")
    print("Presiona Ctrl+C para detener el programa.")

    while True:
        try:
            # Esperar confirmación del usuario
            input("Presiona Enter para iniciar la grabación...")

            # Configuración de grabación
            duration = 4  # Duración en segundos
            sample_rate = 22050  # Tasa de muestreo

            print("Grabando...")
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()  # Esperar a que termine la grabación
            print("Grabación finalizada.")

            # Preguntar al usuario si el audio es de un disparo
            is_gunshot = input("¿Es un disparo? (s/n): ").strip().lower()
            while is_gunshot not in ["s", "n"]:
                is_gunshot = input("Entrada inválida. ¿Es un disparo? (s/n): ").strip().lower()

            label = "1" if is_gunshot == "s" else "0"

            # Generar nombre único para el archivo
            timestamp = int(time.time())
            audio_filename = f"audio_{timestamp}.wav"
            audio_filepath = os.path.join(folder_name, audio_filename)

            # Guardar archivo WAV
            write(audio_filepath, sample_rate, audio_data)
            print(f"Archivo guardado: {audio_filepath}")

            # Registrar información en el archivo CSV
            with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([audio_filename, label])

            print(f"Información registrada en: {csv_file}\n")

        except KeyboardInterrupt:
            print("\nPrograma detenido por el usuario.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
