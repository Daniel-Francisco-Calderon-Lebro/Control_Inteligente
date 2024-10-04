# leo la base de datosdesde el excel descarto el encabezado

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

data=np.loadtxt(r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Control Inteligente\Parcial1\Base datos parcial.txt',delimiter='\t',skiprows=1)#llamar la base de datos
Tiempo=data[:,0].T
voltaje_entrada=data[:,1].T
Corriente_salida=data[:,2].T 

print(Corriente_salida)

plt.plot(Tiempo,voltaje_entrada)
plt.plot(Tiempo,Corriente_salida)
plt.show()

# Creo archivo Base de datos Real
filename='data_Simulada_Parcial.txt'
data = np.vstack((Tiempo,voltaje_entrada,Corriente_salida)).T
top = 'Time (sec), Heater 1 (%), Temperature 1 (degC)'
np.savetxt(filename, data, delimiter=',', header=top, comments='')


import numpy as np
import matplotlib.pyplot as plt

def normalizar(vector):
    min_val = np.min(vector)
    max_val = np.max(vector)
    
    for i in range(len(vector)):
        vector[i] = (vector[i] - min_val) / (max_val - min_val)
    
    return vector, min_val, max_val  

# Cargar el archivo .txt
data = np.loadtxt(r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Control Inteligente\Parcial1\data_Simulada_Parcial.txt', delimiter=',', skiprows=1)
Tiempo_data = data[:, 0]
Escalon_data = data[:, 1]
Salida_data = data[:, 2]

# Normalizar los datos
tiempo_normalizado, min_tiempo, max_tiempo = normalizar(Tiempo_data)
escalon_normalizado, min_escalon, max_escalon = normalizar(Escalon_data)
salida_normalizado, min_salida, max_salida = normalizar(Salida_data)

print(f"min_tiempo: {min_tiempo}, max_tiempo: {max_tiempo}")
print(f"min_escalon: {min_escalon}, max_escalon: {max_escalon}")
print(f"min_salida: {min_salida}, max_salida: {max_salida}")







# creo un archivo .txt con los datos normalizados
filename='data_normalizada.txt'
data = np.vstack((tiempo_normalizado,escalon_normalizado,salida_normalizado)).T
top = 'Time (sec), Heater 1 (%), Temperature 1 (degC)'
np.savetxt(filename, data, delimiter=',', header=top, comments='')


####################################################generrador de retrasos####################################################

kr =  2# Cantidad de retrasos a implementar

# Función para crear vectores desplazados
def crear_vectores_desplazados(vector, num_retrasos):
    vectores = []
    n = len(vector)-1
    
    for i in range(num_retrasos):
        # Desplazar el vector hacia atrás
        # Calcular los índices de inicio y fin
        start_index = i
        end_index = n - (num_retrasos - 1) + i
        # Asegurarse de que el segmento tenga el tamaño correcto
        if end_index <= n:
            resultado = vector[start_index:end_index]
            # Añadir padding si es necesario
            if len(resultado) < (n - (num_retrasos - 1)):
                resultado = np.pad(resultado, (0, (n - (num_retrasos - 1)) - len(resultado)), mode='constant', constant_values=0)
            vectores.append(resultado)
    
    return vectores



#Cargar el archivo .txt
data = np.loadtxt(r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Control Inteligente\Parcial1\data_normalizada.txt', delimiter=',', skiprows=1)
Tiempo_normalizado_data = data[:, 0]
Escalon_normalizado_data = data[:, 1]
Salida_normalizado_data = data[:, 2]

print(Tiempo_normalizado_data.shape, Escalon_normalizado_data.shape, Salida_normalizado_data.shape)

# yp[k] = f(yp[k-1], yp[k-2],..., yp[k-N], u[k-1], u[k-2],..., u[k-N])}


# Datos de ejemplo
x = np.copy(Escalon_normalizado_data)
y = np.copy(Salida_normalizado_data)

# Número de retrasos


# Crear vectores desplazados
vectores_desplazados_Escalon = crear_vectores_desplazados(x, kr)
vectores_desplazados_Salida = crear_vectores_desplazados(y, kr)

# Imprimir las formas de los vectores desplazados
print((vectores_desplazados_Escalon[0]).shape)
print((vectores_desplazados_Salida[0]).shape)
print((y[kr:].shape))
print((vectores_desplazados_Escalon))
print((vectores_desplazados_Salida))
print((y[kr:]))

# Crear archivo .txt con los vectores desplazados
filename='vectores_desplazados.txt'
data = np.vstack((vectores_desplazados_Escalon,vectores_desplazados_Salida,y[kr:])).T
#organizo el top
top = 'U-2, U-1,..................,U-n     Y-2, Y-1,................,Y-n,.................Yk'
np.savetxt(filename, data, delimiter=',', header=top, comments='')



####################################################Entrenador de red####################################################

data = np.loadtxt(r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Control Inteligente\Parcial1\vectores_desplazados.txt', delimiter=',', skiprows=1)

# Función para separar X1, X2, y y hallar el numero de retrasos
k = (int(len(data[0])/2))
def seprarar_X1_X2_y(data,k):
    X1 = data[:, :k-1]
    X2 = data[:, k:2*k]
    y = data[:, 2*k:]
    X = np.concatenate((X1, X2), axis=1)
    return X,y

X,y = seprarar_X1_X2_y(data,k)
print(data.shape)

# Crear archivo .txt con los vectores desplazados
filename='Datos_Entrada_red.txt'
data = np.vstack((X))
#organizo el top
top = 'U-2, U-1,..................,U-n     Y-2, Y-1,................,Y-n,.................Yk'
np.savetxt(filename, data, delimiter=',', header=top, comments='')




# ######################Modelo Funciona Perfecto###################
# Crear el modelo
model = Sequential()
model.add(Dense(2, input_dim=(k*2)-1, activation='relu'))
model.add(Dense(1,activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_squared_error'])  # métrica a MSE


# 4. Entrenamos la red
model.fit(X, y, epochs=500, verbose=2)

# 5. Evaluamos la red
scores = model.evaluate(X, y)

# 6. Mostramos los resultados
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# 7. Mostramos la red
print(model.predict(X))

# Guardar el modelo
model.save('Modelo_entrenado_parcial.h5')

######################Modelo Funciona###################


model = load_model(r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Control Inteligente\Parcial1\Modelo_entrenado_parcial.h5')
data = np.loadtxt(r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Control Inteligente\Parcial1\vectores_desplazados.txt', delimiter=',', skiprows=1)

# Función para separar X1, X2, y y hallar el numero de retrasos
k = (int(len(data[0])/2))
def seprarar_X1_X2_y(data,k):
    X1 = data[:, :k-1]
    X2 = data[:, k:2*k]
    y = data[:, 2*k:]
    X = np.concatenate((X1, X2), axis=1)
    return X,y

X,y = seprarar_X1_X2_y(data,k)

print(X.shape)

# Evaluamos la red
scores = model.evaluate(X, y)

print(f'Este es el score: {scores}')

# Mostramos las predicciones
predictions = model.predict(X)


##############################importante agregar el valor maxima de las salidas normalizadas################################
def desnormalizar(predicciones, min_val, max_val):
    return predicciones * (max_val - min_val) + min_val

valor_maximo = 0.144475327
predictions = desnormalizar(predictions, 0, valor_maximo) # valores reales
y = desnormalizar(y, 0, valor_maximo) # valores reales


##############################################importante agregar el valor maxima de las salidas normalizadas################################

# Graficar las predicciones frente a los valores reales
plt.figure(figsize=(10, 6))
plt.plot(y, label='Valores reales', color='blue', linewidth=2)
plt.plot(predictions, label='Predicciones', color='red', linestyle='dashed')
plt.title('Comparación de predicciones vs valores reales')
plt.xlabel('Índice de muestra')
plt.ylabel('Salida')
plt.legend()
plt.show()