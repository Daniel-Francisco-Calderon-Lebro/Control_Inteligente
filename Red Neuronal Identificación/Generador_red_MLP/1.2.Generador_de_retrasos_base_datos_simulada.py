import numpy as np

kr = 2 # Cantidad de retrasos a implementar

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
data = np.loadtxt(r'Red Neuronal Identificación\Generador_red_MLP\data_normalizada.txt', delimiter=',', skiprows=1)
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