#Este codigo lo que hace es que prueba los modelos que quieras con audios pre hechos en "generate_test_audio.py"

import os
import csv
import numpy as np
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Cargar los modelos entrenados
model = load_model('Models Save IA/GunshotIaModel_v3.h5')
model2 = load_model("Models Save IA/GunshotIaModel_v2_1.h5")

# Ruta de la carpeta con los audios y el archivo CSV
folder_name = "Test audio IA/Test Automatic/testing"
csv_file = os.path.join(folder_name, "labels.csv")

def load_audio(file_path, sample_rate=44100, duration=4):
    """
    Carga un archivo de audio y ajusta su duración.

    :param file_path: Ruta del archivo de audio.
    :param sample_rate: Frecuencia de muestreo del audio.
    :param duration: Duración deseada del audio en segundos.
    :return: Audio como array de numpy.
    """
    audio, sr = librosa.load(file_path, sr=sample_rate)
    num_samples = duration * sample_rate
    if len(audio) > num_samples:
        audio = audio[:num_samples]
    else:
        silence = np.zeros(num_samples - len(audio))
        audio = np.concatenate([audio, silence])
    return audio

def process_audio(audio, sample_rate=22050):
    """
    Genera un espectrograma a partir del audio.

    :param audio: Audio como array de numpy.
    :param sample_rate: Frecuencia de muestreo del audio.
    :return: Espectrograma del audio procesado.
    """
    spect = librosa.stft(audio)
    spect_db = librosa.amplitude_to_db(spect, ref=np.max)
    spect_db = np.expand_dims(spect_db, axis=-1)  # Agregar una dimensión para el canal
    return spect_db

def predict_gunshot(spectrogram):
    """
    Realiza la predicción con los modelos entrenados.

    :param spectrogram: Espectrograma del audio.
    :return: Predicciones de probabilidad de ambos modelos.
    """
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Batch dimension
    prob = model.predict(spectrogram)[0][0]
    prob2 = model2.predict(spectrogram)[0][0]
    return prob, prob2

def evaluate_predictions():
    """
    Evalúa las muestras de audio usando los modelos y compara los resultados
    con las etiquetas definidas en el archivo CSV.
    """
    if not os.path.exists(csv_file):
        print(f"El archivo CSV {csv_file} no existe.")
        return

    y_true = []
    y_pred_model1 = []
    y_pred_model2 = []
    y_pred_modelC = []
    true1 = 0
    true2 = 0
    trueC = 0
    with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            file_path = os.path.join(folder_name, row['filename'])
            is_gunshot = int(row['is_gunshot'])

            if not os.path.isfile(file_path):
                print(f"El archivo {file_path} no se encuentra.")
                continue

            # Cargar y procesar el audio
            audio = load_audio(file_path)
            audio_resampled = librosa.resample(audio, orig_sr=44100, target_sr=22050)
            spectrogram = process_audio(audio_resampled)

            # Realizar predicciones
            prob, prob2 = predict_gunshot(spectrogram)

            # Convertir probabilidades en etiquetas
            pred1 = 1 if prob > 0.62 else 0
            pred2 = 1 if prob2 > 0.620 else 0

            y_true.append(is_gunshot)
            y_pred_model1.append(pred1)
            y_pred_model2.append(pred2)
            trueComb = 1 if (pred1  or pred2) else 0
            y_pred_modelC.append(trueComb)
            print(f"Archivo: {row['filename']}")
            print(f"Etiqueta real: {'Disparo' if is_gunshot else 'No disparo'}")
            print(f"Modelo 1: {prob:.3f} {'Disparo' if pred1 else 'No disparo'}")
            print(f"Modelo 2: {prob2:.3f} {'Disparo' if pred2 else 'No disparo'}")
            print("-------------------------------------------")
            true1 = true1 + 1 if pred1 == is_gunshot else true1
            true2 = true2 + 1 if pred2 == is_gunshot else true2
            trueC = trueC + 1 if (pred1 == is_gunshot or pred2 == is_gunshot) else trueC
    print('modelo 1 aciertos: ',true1)
    print('modelo 2 aciertos',true2)
    print('modelo combinado aciertos',trueC)
    # Calcular y mostrar matrices de confusión
    cm1 = confusion_matrix(y_true, y_pred_model1, labels=[0, 1])
    cm2 = confusion_matrix(y_true, y_pred_model2, labels=[0, 1])
    cm3 = confusion_matrix(y_true, y_pred_modelC, labels=[0, 1])
    print("Matriz de confusión - Modelo 1:")
    print(cm1)
    print("Matriz de confusión - Modelo 2:")
    print(cm2)
    print("Matriz de confusión - Modelo 3:")
    print(cm3)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=["No disparo", "Disparo"])
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=["No disparo", "Disparo"])

    disp1.plot(cmap="Blues")
    plt.title("Matriz de Confusión - Modelo 1")
    plt.show()

    disp2.plot(cmap="Greens")
    plt.title("Matriz de Confusión - Modelo 2")
    plt.show()

if __name__ == "__main__":
    evaluate_predictions()
