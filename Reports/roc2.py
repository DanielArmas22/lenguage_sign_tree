from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import h5py
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Cargar el modelo y el encoder de etiquetas
model = joblib.load('./model/modelo.joblib')
encoder = joblib.load('./model/labels.joblib')

# Cargar los datos de prueba
with h5py.File('./points/data.h5', 'r') as f:
    data = f['data'][:]
    labels = f['labels'][:]

data = np.asarray(data)
labels = np.asarray(labels)

# Decodificar etiquetas si están en bytes
labels = [
    [label.decode('utf-8') if isinstance(label, bytes) else label for label in sample]
    for sample in labels
]

# Dividir los datos (considera usar una técnica de estratificación adecuada para multietiquetas)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True
    # stratify=labels  # Podría no ser adecuado para multietiquetas
)

# Binarizar las etiquetas usando solo y_train para fit
mlb = MultiLabelBinarizer()
y_train_binarized = mlb.fit_transform(y_train)
y_test_binarized = mlb.transform(y_test)
classes = mlb.classes_

# Obtener probabilidades de predicción
y_proba = model.predict_proba(x_test)

# Verificar que y_proba tenga la misma cantidad de columnas que clases
# assert y_proba.shape[1] == len(classes), "Las probabilidades predichas no coinciden con el número de clases."

selected_label = '-'  # Cambia esto según necesites

if selected_label != '-':
    if selected_label in classes:
        label_index = np.where(classes == selected_label)[0][0]
        y_true_label = y_test_binarized[:, label_index]
        y_proba_label = y_proba[:, label_index]

        # Calcular la Curva ROC y el AUC para la etiqueta seleccionada
        fpr, tpr, thresholds = roc_curve(y_true_label, y_proba_label)
        auc_score = roc_auc_score(y_true_label, y_proba_label)

        # Graficar la Curva ROC
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'{selected_label} (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Azar')
        plt.xlabel('Tasa de Falsos Positivos (FPR)')
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
        plt.title(f'Curva ROC para el Label: {selected_label}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
        print(f"AUC para la clase '{selected_label}': {auc_score:.2f}")
    else:
        print(f"El label seleccionado '{selected_label}' no existe en las clases.")
else:
    auc_scores = {}
    plt.figure()
    for i, class_label in enumerate(classes):
        y_true = y_test_binarized[:, i]
        y_proba_class = y_proba[:, i]
        
        # Verificar si hay suficientes clases positivas y negativas
        if np.unique(y_true).size < 2:
            print(f"Clase '{class_label}' no tiene suficientes muestras para calcular ROC.")
            continue
        
        fpr, tpr, _ = roc_curve(y_true, y_proba_class)
        auc_score = roc_auc_score(y_true, y_proba_class)
        auc_scores[class_label] = auc_score
        plt.plot(fpr, tpr, label=f'{class_label} (AUC = {auc_score:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Azar')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curvas ROC para Clasificación Multietiqueta')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    print("AUC Scores por Clase:", auc_scores)
