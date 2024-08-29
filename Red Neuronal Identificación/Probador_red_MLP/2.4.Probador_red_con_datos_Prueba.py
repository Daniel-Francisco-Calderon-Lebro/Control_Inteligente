import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt




# Cargar el modelo
model = load_model(r'Red Neuronal Identificación\Generador_red_MLP\Modelo_entrenado.h5')
data = np.loadtxt(r'Red Neuronal Identificación\Probador_red_MLP\vectores_desplazados_prueba.txt', delimiter=',', skiprows=1)

# Función para separar X1, X2, y y hallar el numero de retrasos
k = (int(len(data[0])/2))

X = data[:, :k]

print(X[0].shape)

# Mostramos las predicciones
prediction = model.predict([[0, 0, 0, 0]])

print(prediction)
'''

##############################importante agregar el valor maxima de las salidas normalizadas################################
def desnormalizar(predicciones, min_val, max_val):
    return predicciones * (max_val - min_val) + min_val

valor_maximo = 9.59999663472868
predictions = desnormalizar(predictions, 0, valor_maximo) # valores reales
y = desnormalizar(y, 0, valor_maximo) # valores reales


##############################################importante agregar el valor maxima de las salidas normalizadas################################

# Graficar las predicciones frente a los valores reales
plt.figure(figsize=(10, 6))
plt.plot(y, label='Valores reales', color='blue', linewidth=2)
plt.plot(predictions, label='Predicciones', color='red', linestyle='dashed')
plt.plot(y-predictions, label='Error', color='green', linestyle='dashed')
plt.title('Comparación de predicciones vs valores reales y error')
plt.xlabel('Índice de muestra')
plt.ylabel('Salida')
plt.legend()
plt.show()

'''