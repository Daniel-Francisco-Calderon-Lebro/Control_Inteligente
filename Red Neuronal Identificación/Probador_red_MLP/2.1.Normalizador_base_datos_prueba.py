import numpy as np
import matplotlib.pyplot as plt

def normalizar(vector, min_val=None, max_val=None):
    for i in range(len(vector)):
        vector[i] = (vector[i] - min_val) / (max_val - min_val)
    
    return vector, min_val, max_val  

# Cargar el archivo .txt
data = np.loadtxt(r'Red Neuronal Identificación\Probador_red_MLP\data_prueba_red.txt', delimiter=',', skiprows=1)
Tiempo_data = data[:, 0]
Escalon_data = data[:, 1]
Salida_data = data[:, 2]

# Normalizar los datos
tiempo_normalizado, min_tiempo, max_tiempo = normalizar(Tiempo_data,0,1)#no tener en cuenta este valor no se usa
escalon_normalizado, min_escalon, max_escalon = normalizar(Escalon_data,0,5) #cambiar este valor al maximo valor de la entrada normalizada de datos de entrenaamiento de la red
salida_normalizado, min_salida, max_salida = normalizar(Salida_data,0,5.999999326945357)#cambiar este valor al maximo valor de la salida normalizada de datos de entrenaamiento de la red

print(f"min_tiempo: {min_tiempo}, max_tiempo: {max_tiempo}")
print(f"min_escalon: {min_escalon}, max_escalon: {max_escalon}")
print(f"min_salida: {min_salida}, max_salida: {max_salida}")

# creo un archivo .txt con los datos normalizados
filename='data_normalizada_prueba.txt'
data = np.vstack((tiempo_normalizado,escalon_normalizado,salida_normalizado)).T
top = 'Time (sec), Heater 1 (%), Temperature 1 (degC)'
np.savetxt(filename, data, delimiter=',', header=top, comments='')


# yp[k] = f(yp[k-1], yp[k-2],..., yp[k-N], u[k-1], u[k-2],..., u[k-N])}

