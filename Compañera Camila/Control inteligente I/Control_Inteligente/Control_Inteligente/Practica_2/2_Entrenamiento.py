# Practica 1 : Entrenamiento Red Neuronal Simple

# Librerias 
import numpy as np
import keras 
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm 
import pandas as pd

# Leer Datos de Exel
data =pd.read_excel('C:/Users/Jhon Willian/Documents/Visual/Practica_2/data_1.xlsx')

# Matriz 1170 x 4
colum = ['uk_1','uk_2','yk_1','yk_2']
fila = 1170

select_data = data[colum].iloc[:fila]

# Matriz de Entrada (In)
in_colum = ['uk_1','uk_2','yk_1','yk_2']
training_data = data[in_colum].iloc[:fila].values

# Vector de Salida (Out)
out_colum = 'yk'
target_data = data[out_colum].iloc[:fila].values

# Inicio arquitectura de red neuronal 
model = Sequential()

# Capas: Estructura: model.add(Dense(# Neuronas (capa), # Entradas, Funcion de activacion))

model.add(Dense(4, input_dim=4, activation='relu'))     # Capa Oculta 1
model.add(Dense(8, activation='relu'))                 # Capa Oculta 2
model.add(Dense(1, activation='relu'))              # Capa de Salida

# Argumentos para el Aprendizaje
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])
 
# Entrenamiento red neuronal
model.fit(training_data, target_data, epochs=100)
 
# Evaluacion red neuronal
scores = model.evaluate(training_data, target_data)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print (model.predict(training_data).round())

# Modelo Encontrado
modelo = model.predict(training_data)

# Error Cuadratico Medio
mse = (np.square(target_data - modelo)).mean()
print(f"MSE {mse}")


# Grafico
plt.grid()
plt.title('Datos vs Modelo')
plt.plot(target_data,'b-',linewidth=3)
plt.plot(modelo,'r--',linewidth=2)
plt.legend(['Datos ', 'Modelo '],loc = "lower right")
plt.ylabel(' Y(K)',fontsize=14)
plt.xlabel('Tiempo (s)',fontsize=14)
plt.show()

# Salvar Modelo

# Serializar el modelo a JSON
model_json = model.to_json()
with open("model_03.json", "w") as json_file:
    json_file.write(model_json)

# Serializar los pesos a HDF5
model.save_weights("model_03.h5")
print("Modelo Guardado!")
