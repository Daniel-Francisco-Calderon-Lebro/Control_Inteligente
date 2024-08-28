import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import pandas as pd

# Cargar el modelo
model = load_model(r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Control Inteligente\Red Neuronal Identificación\Modelo_identificacion.h5')
data = np.loadtxt(r'Red Neuronal Identificación\vectores_desplazados.txt', delimiter=',', skiprows=1)

# Función para separar X1, X2, y y hallar el numero de retrasos
k = (int(len(data[0])/2))
def seprarar_X1_X2_y(data,k):
    X1 = data[:, :k]
    X2 = data[:, k:2*k]
    y = data[:, 2*k:]
    X = np.concatenate((X1, X2), axis=1)
    return X,y

X,y = seprarar_X1_X2_y(data,k)
print(data.shape)

# Evaluamos la red
scores = model.evaluate(X, y)

# Mostramos las predicciones
predictions = model.predict(X)

def desnormalizar(predicciones, min_val, max_val):
    return predicciones * (max_val - min_val) + min_val

predictions = desnormalizar(predictions, 0, 5.999999326945357) # valores reales
y = desnormalizar(y, 0, 5.999999326945357) # valores reales


# Graficar las predicciones frente a los valores reales
plt.figure(figsize=(10, 6))
plt.plot(y, label='Valores reales', color='blue', linewidth=2)
plt.plot(predictions, label='Predicciones', color='red', linestyle='dashed')
plt.title('Comparación de predicciones vs valores reales')
plt.xlabel('Índice de muestra')
plt.ylabel('Salida')
plt.legend()
plt.show()