import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Leer Datos de Exel
data = pd.read_excel("C:/Users/Jhon Willian/Documents/Visual/Practica_4/Datos/DATA.xlsx")

# Matriz n x 4
selected_columns = ["uk_1", "uk_2", "yk_1", "yk_2", "yk"]
selected_rows = 8000

selected_data = data[selected_columns].iloc[:selected_rows]

# Matriz de Entrada (In´s)
input_columns = ["uk_1", "uk_2", "yk_1", "yk_2"]
training_data = data[input_columns].iloc[:selected_rows].values

# Vector de Salida (Out)
output_column = "Salida"
target_data = data[output_column].iloc[:selected_rows].values

# Vector de Entrada (IN)
in_column = "Entrada"
target_data_1 = data[in_column].iloc[:selected_rows].values

# Inicio arquitectura de red neuronal 
model = tf.keras.Sequential()

# Primera Capa: Estructura: model.add(Dense(# Neuronas (capa), # Entradas, Funcion de activacion))
# 2 Capas: Estructura: model.add(Dense(# Neuronas (capa), Funcion de activacion))

model.add(tf.keras.layers.Dense(2, input_dim=4, activation='tanh'))
model.add(tf.keras.layers.Dense(2, activation='tanh'))
model.add(tf.keras.layers.Dense(1, activation='tanh'))

# Argumentos para el Aprendizaje
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

# Entrenamiento red neuronal
model.fit(training_data, target_data, epochs=500)

# Evaluamos el modelo
scores = model.evaluate(training_data, target_data)

# Modelo Encontrado
modelo = model.predict(training_data)

# Predicciones
print("Predicciones:")
print(modelo)

# Error Cuadratico Medio
mse = (np.square(target_data - modelo)).mean()
print(f"MSE {mse}")

plt.grid()
plt.title('Entrenamiento: Datos vs Modelo')
plt.plot(target_data,'b-',linewidth=3)
plt.plot(modelo,'r--',linewidth=2)
plt.legend(['Datos ', 'Modelo '],loc = "lower right")
plt.ylabel(' Y(K)',fontsize=14)
plt.xlabel('Tiempo (s)',fontsize=14)
plt.show()

# Salvar Modelo

# Serializar el modelo a JSON
model_json = model.to_json()
with open("model_Tanh.json", "w") as json_file:
    json_file.write(model_json)
# Serializar los pesos a H5
model.save_weights("model_Tanh.h5")
print("Modelo Guardado!")

