import numpy as np
import matplotlib.pyplot as plt

def normalizar(vector):
    for i in range(len(vector)):
        vector[i] = (vector[i] - np.min(vector)) / (np.max(vector) - np.min(vector))

    return vector   

def nueva_tabla(x, y, k):
    u = np.zeros(len(x)-k)
    z = np.zeros(len(x)-k)

    for i in range(len(x)-k):
        u[i:0] = x[i]
        z[i:0] = y[i]
    
    print(u.T, z.T)

# Cargar el archivo .txt
data = np.loadtxt(r'Red Neuronal Identificación\data.txt', delimiter=',', skiprows=1)
Tiempo_data = data[:, 0]
Escalon_data = data[:, 1]
Salida_data = data[:, 2]

# Normalizar los datos
tiempo_normalizado = normalizar(Tiempo_data)
escalon_normalizado = normalizar(Escalon_data)
salida_normalizado = normalizar(Salida_data)

# print(tiempo_normalizado)
# print(Escalon_data)
# print(escalon_normalizado)

# creo un archivo .txt con los datos normalizados
filename='data_normalizada.txt'
data = np.vstack((tiempo_normalizado,escalon_normalizado,salida_normalizado)).T
top = 'Time (sec), Heater 1 (%), Temperature 1 (degC)'
np.savetxt(filename, data, delimiter=',', header=top, comments='')


# yp[k] = f(yp[k-1], yp[k-2],..., yp[k-N], u[k-1], u[k-2],..., u[k-N])}

nueva_tabla(escalon_normalizado[0:9], salida_normalizado[0:9], 2)