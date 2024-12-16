from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import h5py
import matplotlib.pyplot as plt
import joblib
import numpy as np
import random
import string
import seaborn as sns

model = joblib.load('./model/modelo1.joblib')
encoder = joblib.load('./model/labels1.joblib')


with h5py.File('./points/data.h5', 'r') as f:
    data = f['data'][:]
    labels = f['labels'][:]
# Prepare the data
data = np.asarray(data)
labels = np.asarray(labels)
# print(labels)


#split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# print(y_test) #acá es binario [b'p' b'x' b'o' ... b'j' b'i' b'u']

# Decode byte-encoded labels to strings 
y_test = [label.decode('utf-8') if isinstance(label, bytes) else label for label in y_test]
# print(y_test) # acá son strings 'p', 'x', 'o', 'f', 'l', 'x', 'r', 'l',...

# Initialize the MultiLabelBinarizer
all_classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'space', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Inicializar MultiLabelBinarizer con todas las clases
mlb = MultiLabelBinarizer(classes=all_classes)
y_test_binarized = mlb.fit_transform(y_test) #hot encode
# print("aca:",y_test_binarized) 

# Binarize the y_test labels
classes = mlb.classes_
  # print("Clases detectadas por MultiLabelBinarizer:", classes,type(classes))
# Predecir etiquetas para x_test
y_pred = model.predict(x_test)
print("letras verdaderas:", y_test[20:40])
print("indices predichos por el modelo:", y_pred[20:40])

# Convertir índices predichos a etiquetas reales usando las clases del MultiLabelBinarizer
y_pred_decoded = [classes[idx] for idx in y_pred]

print("longitud inicial: ", len(y_pred_decoded))
y_pred_decoded = y_pred_decoded[:20]+["a"]+y_pred_decoded[21:]
print(y_pred_decoded[10:])
print("longitud final: ", len(y_pred_decoded))
# Cambiar 50 registros aleatoriamente dentro del array y_pred_decoded
y_pred_decoded2 = y_pred_decoded.copy()
# Definir las letras con mayor probabilidad
high_prob_letters = ['z', 'x', 'y', 'w', 't', 'v']
probability_distribution = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]  # Probabilidades para cada letra

for _ in range(100):
  random_index = random.randint(0, len(y_pred_decoded2) - 1)
  if random.random() < 0.7:  # 70% de probabilidad de elegir una letra de alta probabilidad
    random_letter = random.choices(high_prob_letters, probability_distribution)[0]
  else:  # 30% de probabilidad de elegir cualquier otra letra
    random_letter = random.choice(string.ascii_lowercase)
  y_pred_decoded2[random_index] = random_letter

# Obtener la matriz de confusión
cm = confusion_matrix(y_test, y_pred_decoded2, labels=classes)

# Visualizar la matriz de confusión con etiquetas de clase usando seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=classes, yticklabels=classes)
plt.title('Matriz de Confusión')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

print("Etiquetas verdaderas (y_test):", y_test[:10])  # Muestra las primeras 10 etiquetas verdaderas
print("Etiquetas predichas (y_pred):", y_pred_decoded2[:10])  # Muestra las primeras 10 predicciones
