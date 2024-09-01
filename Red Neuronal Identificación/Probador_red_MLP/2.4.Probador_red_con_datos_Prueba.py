import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error



# Cargar el modelo
model = load_model(r'Red Neuronal Identificación\Generador_red_MLP\Modelo_entrenado.h5')
data = np.loadtxt(r'Red Neuronal Identificación\Probador_red_MLP\vectores_desplazados_prueba.txt', delimiter=',', skiprows=1)

# Función para separar X y hallar el número de retrasos (k)
k = int(len(data[0]) / 2)  # Este valor varía dependiendo de la cantidad de datos
X = data[:, :k]
y = data[:, -1]  # Usar la última columna como los valores reales

# Iniciar con ceros
nuevafila_inicial = np.zeros((1, X.shape[1]))
X_conceroinicial = np.vstack((nuevafila_inicial, X))

# Inicializar todos los Yk como arrays de longitud k
Yk_retrasos = np.zeros((k,))  # Vector con `k` ceros para los valores Yk

# Lista para almacenar las entradas y predicciones
entradas_y_predicciones = []

# Realizar las predicciones iterativamente
for i in range(X_conceroinicial.shape[0]):
    # Concatenar X[i] con todos los valores de Yk (Yk_k, Yk_k-1, ..., Yk_1)
    X_Entrada_i = np.hstack((X_conceroinicial[i], Yk_retrasos))
    X_Entrada_i = X_Entrada_i.reshape(1, -1)
    
    # Hacer la predicción
    prediccion = model.predict(X_Entrada_i).flatten()[0]
    
    # Almacenar la entrada y la predicción en la lista
    entradas_y_predicciones.append(np.hstack((X_Entrada_i.flatten(), prediccion)))
    
    # Actualizar los valores de Yk: desplazamos los valores hacia atrás y colocamos la nueva predicción
    Yk_retrasos = np.roll(Yk_retrasos, -1)
    Yk_retrasos[-1] = prediccion

# Convertir la lista en un array de NumPy
entradas_y_predicciones = np.array(entradas_y_predicciones)

# Crear el archivo .txt con los vectores desplazados
filename = 'Valores_de_Pruebared.txt'

# Definir el encabezado basado en el número de entradas (U) y retardos (Yk)
encabezado = f"{', '.join([f'U-{i+1}' for i in range(k)])}, " \
             f"{', '.join([f'Y-{i+1}' for i in range(k, 0, -1)])}, Yk"

# Guardar los datos en un archivo .txt
np.savetxt(filename, entradas_y_predicciones, delimiter=',', header=encabezado, comments='', fmt='%.6e')

print(f"Archivo '{filename}' guardado exitosamente.")




############################## Importante agregar el valor máximo de las salidas normalizadas ##############################
def desnormalizar(predicciones, min_val, max_val):
    return predicciones * (max_val - min_val) + min_val

# Desnormalizar las predicciones y los valores reales
valor_maximo = 5.999999326945357
prediction_flat = desnormalizar(entradas_y_predicciones[:, -1], 0, valor_maximo)  # Aplicar desnormalización
y = desnormalizar(y, 0, valor_maximo)

############################## Fin de la desnormalización ##############################
prediction_flat = prediction_flat[:-1] # Quitar la ultima fila de ceros

# Calcular métricas
mse = mean_squared_error(y, prediction_flat)
mae = mean_absolute_error(y, prediction_flat)

print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")

# Graficar las predicciones frente a los valores reales
plt.figure(figsize=(12, 8))

# Graficar valores reales, predicciones y error
plt.plot(y, label='Valores reales', color='blue', linewidth=2)
plt.plot(prediction_flat, label='Predicciones', color='green', linestyle='dashed')
plt.plot(y - prediction_flat, label='Error', color='red', linestyle='dashed')

# Añadir título con las métricas
plt.title(f'Comparación de predicciones vs valores reales y error\n'
          f'MSE: {mse:.6f} | MAE: {mae:.6f}', fontsize=14)

plt.xlabel('Índice de muestra', fontsize=12)
plt.ylabel('Salida', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()







































































































# import numpy as np
# from keras.models import load_model
# import matplotlib.pyplot as plt

# # Cargar el modelo
# model = load_model(r'Red Neuronal Identificación\Generador_red_MLP\Modelo_entrenado.h5')
# data = np.loadtxt(r'Red Neuronal Identificación\Probador_red_MLP\vectores_desplazados_prueba.txt', delimiter=',', skiprows=1)

# # Función para separar X1, X2, y y hallar el numero de retrasos
# k = (int(len(data[0])/2))

# X = data[:, :k]

# Yk_2 = []
# Yk_1 = []
# Yk = []

# # 1. Iniciamos el primer Vector en cero
# nuevafila_inicial=np.zeros((1, X.shape[1]))
# X_conceroinicial=np.vstack((nuevafila_inicial, X))


# # 2. Iniciamos yks con ceros para el primer instante
# Yk_2_1 = np.zeros(1)  # Fila de cero para Yk_2
# Yk_1_1 = np.zeros(1)  # Fila de cero para Yk_1

# # 3. Hacemos el primer instante de la respuesta de la red
# X_Entrada_i_0 = np.hstack((X_conceroinicial[0], Yk_2_1, Yk_1_1))
# primer_valor = X_Entrada_i_0.reshape(1, -1)
# Valor0=model.predict(primer_valor)

# # 4. Pasamos el primer valor de la red al primer retraso
# Yk_2_2 = np.zeros(1)  # Fila de cero para Yk_2
# Yk_1_2 = np.array(Valor0)  # Fila de cero para Yk_1

# # 5. Hacemos el segundo instante de la respuesta de la red
# X_Entrada_i_1 = np.hstack((X_conceroinicial[1], Yk_2_2, Yk_1_2))
# segundo_valor = X_Entrada_i_1.reshape(1, -1)
# Valor1=model.predict(segundo_valor)

# # 6. Pasamos el primer valor de la red al segundo retraso y el segundo valor de la red al primer retraso
# Yk_2_3 = np.array(Valor0)  # Fila de cero para Yk_2
# Yk_1_3 = np.array(Valor1)  # Fila de cero para Yk_1

# # 7. Hacemos el tercer instante de la respuesta de la red
# X_Entrada_i_2 = np.hstack((X_conceroinicial[2], Yk_2_3, Yk_1_3))
# tercer_valor = X_Entrada_i_2.reshape(1, -1)
# Valor2=model.predict(tercer_valor)

# # 8. Pasamos el primer valor de la red al tercer retraso si lo hubiera y el segundo valor de la red al segundo retraso
# Yk_2_4 = np.array(Valor1)  # Fila de cero para Yk_2
# Yk_1_4 = np.array(Valor2)  # Fila de cero para Yk_1

# # 9. Hacemos el cuarto instante de la respuesta de la red
# X_Entrada_i_3 = np.hstack((X_conceroinicial[3], Yk_2_4, Yk_1_4))
# cuarto_valor = X_Entrada_i_3.reshape(1, -1)
# Valor3=model.predict(cuarto_valor)

# # 10. Pasamos el primer valor de la red al cuarto retraso si lo hubiera y el segundo valor de la red al segundo retraso
# Yk_2_5 = np.array(Valor2)  # Fila de cero para Yk_2
# Yk_1_5 = np.array(Valor3)  # Fila de cero para Yk_1



































                    




















# ##############################importante agregar el valor maxima de las salidas normalizadas################################
# def desnormalizar(predicciones, min_val, max_val):
#     return predicciones * (max_val - min_val) + min_val

# valor_maximo = 9.59999663472868
# prediction = desnormalizar(prediction, 0, valor_maximo) # valores reales
# y = desnormalizar(y, 0, valor_maximo) # valores reales


# ##############################################importante agregar el valor maxima de las salidas normalizadas################################

# # Graficar las predicciones frente a los valores reales
# plt.figure(figsize=(10, 6))
# plt.plot(y, label='Valores reales', color='blue', linewidth=2)
# plt.plot(predictions, label='Predicciones', color='red', linestyle='dashed')
# plt.plot(y-predictions, label='Error', color='green', linestyle='dashed')
# plt.title('Comparación de predicciones vs valores reales y error')
# plt.xlabel('Índice de muestra')
# plt.ylabel('Salida')
# plt.legend()
# plt.show()


















# import numpy as np
# from keras.models import load_model
# import matplotlib.pyplot as plt

# # Cargar el modelo
# model = load_model(r'Red Neuronal Identificación\Generador_red_MLP\Modelo_entrenado.h5')
# data = np.loadtxt(r'Red Neuronal Identificación\Probador_red_MLP\vectores_desplazados_prueba.txt', delimiter=',', skiprows=1)

# # Función para separar X y y, y hallar el número de retrasos
# k = int(len(data[0]) / 2)  # número de retardos
# X = data[:, :k]  # Usar las primeras 'k' columnas como entradas
# y = data[:, -1]  # Última columna como salida ideal del sistema

# prediction = []
# # Realizar predicciones
# for i in range(X.shape[0]):
#     if i < k:  # Para los primeros k valores, inicializamos con ceros
#         prediction.append([0.0])  # Predicciones iniciales con ceros
#     else:
#         # Crear una lista con las predicciones previas necesarias (k retardos)
#         pred_anteriores = [float(prediction[i-j-1][0]) for j in range(k)]

#         # Crear la entrada para el modelo usando X y los k retardos
#         input_model = np.array([list(X[i]) + pred_anteriores], dtype=np.float64)
        
#         # Predecir usando el modelo
#         pred = model.predict(input_model)
#         prediction.append([pred[0][0]])  # Extraer el valor de la predicción
        
#         # Imprimir las predicciones anteriores utilizadas
#         # print(f"Predicciones anteriores para i={i}: {pred_anteriores}")

# # imprimo y hago un nuevo documento .txt


# ############################## Importante agregar el valor máximo de las salidas normalizadas ##############################
# def desnormalizar(predicciones, min_val, max_val):
#     return predicciones * (max_val - min_val) + min_val

# # Aplanar la lista de predicciones
# prediction_flat = np.array(prediction).flatten()

# # Desnormalizar las predicciones y los valores reales
# valor_maximo = 9.59999663472868
# prediction_flat = desnormalizar(prediction_flat, 0, valor_maximo)  # Aplicar desnormalización
# y = desnormalizar(y, 0, valor_maximo)

# ############################## Fin de la desnormalización ##############################

# # Graficar las predicciones frente a los valores reales
# plt.figure(figsize=(10, 6))
# plt.plot(y, label='Valores reales', color='blue', linewidth=2)
# plt.plot(prediction_flat, label='Predicciones', color='red', linestyle='dashed')
# plt.plot(y - prediction_flat, label='Error', color='green', linestyle='dashed')
# plt.title('Comparación de predicciones vs valores reales y error')
# plt.xlabel('Índice de muestra')
# plt.ylabel('Salida')
# plt.legend()
# plt.show()


# prediction_flat = np.array(prediction).flatten()

# # Verificar que prediction_flat tenga el tamaño adecuado
# # El tamaño de prediction_flat debe coincidir con el número de filas de X
# assert len(prediction_flat) == X.shape[0], "La longitud de las predicciones no coincide con el número de filas en X"

# # Concatenar X y las predicciones para guardar en archivo
# X_and_predictions = np.hstack((X, prediction_flat.reshape(-1, 1)))

# # Crear el encabezado para el archivo
# # Asegúrate de que el número de U corresponda al número de columnas en X
# num_us = X.shape[1]  # Número de columnas en X, representa U-n
# top = ', '.join([f'U-{i}' for i in range(num_us, 0, -1)]) + ', Yk'

# # Guardar en archivo .txt
# filename = 'Valorespruebaresultados.txt'
# np.savetxt(filename, X_and_predictions, delimiter=',', header=top, comments='', fmt='%.6f')

# print(f"Archivo '{filename}' guardado exitosamente.")