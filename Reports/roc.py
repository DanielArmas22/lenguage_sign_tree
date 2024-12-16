from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import h5py
import matplotlib.pyplot as plt
import joblib
import numpy as np

model = joblib.load('./model/modelo.joblib')
encoder = joblib.load('./model/labels.joblib')
#load the test data
with h5py.File('./points/data.h5', 'r') as f:
    data = f['data'][:]
    labels = f['labels'][:]
# Prepare the data
data = np.asarray(data)
labels = np.asarray(labels)

#split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Decode byte-encoded labels to strings
y_test = [label.decode('utf-8') if isinstance(label, bytes) else label for label in y_test]

# Initialize the MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Binarize the y_test labels
y_test_binarized = mlb.fit_transform(y_test)
classes = mlb.classes_

#MAtrix
y_proba = model.predict_proba(x_test)
auc_scores = {}
selected_label = '-'

if(selected_label !='-'):
  label_index = np.where(classes == selected_label)[0][0]   
  y_true_label = y_test_binarized[:, label_index]
  y_proba_label = y_proba[:, label_index]

  # Calcular la Curva ROC y el AUC para el label seleccionado
  fpr, tpr, thresholds = roc_curve(y_true_label, y_proba_label)
  auc_score = roc_auc_score(y_true_label, y_proba_label)
  # Extraer las etiquetas verdaderas y las probabilidades predichas para el label seleccionado
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
  for i, class_label in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
    auc_score = roc_auc_score(y_test_binarized[:, i], y_proba[:, i])
    auc_scores[class_label] = auc_score
    plt.plot(fpr, tpr, label=f'{class_label} (AUC = {auc_score:.2f})')

  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlabel('Tasa de Falsos Positivos (FPR)')
  plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
  plt.title('ROC Curves for Multi-Label Classification')
  plt.legend(loc='lower right')
  plt.show()

  print("AUC Scores per Class:", auc_scores)
  #MAtrix

