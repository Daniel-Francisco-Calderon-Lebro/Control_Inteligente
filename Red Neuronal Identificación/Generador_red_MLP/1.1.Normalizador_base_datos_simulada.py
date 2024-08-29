import numpy as np
import matplotlib.pyplot as plt

def normalizar(vector):
    min_val = np.min(vector)
    max_val = np.max(vector)
    
    for i in range(len(vector)):
        vector[i] = (vector[i] - min_val) / (max_val - min_val)
    
    return vector, min_val, max_val  

# Cargar el archivo .txt
data = np.loadtxt(r'Red Neuronal Identificación\Generador_red_MLP\data.txt', delimiter=',', skiprows=1)
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


# yp[k] = f(yp[k-1], yp[k-2],..., yp[k-N], u[k-1], u[k-2],..., u[k-N])}

