import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error



def main():
    """
    Este programa Realiza la predicción de la salida de 
    una red MLP en base a una tabla ya generada de datos diferentes a los 
    de entrenamiento los vectores de entrada Uk y Uk-n ya estan definidos en la misma
    tabla. Trabaja con datos ya generados, normalizados y retrasados. (con vectores creados)
    Este programa no se puede utilizar para un proceso online.    
    """

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
    # for i in range(10):
        # Concatenar X[i] con todos los valores de Yk (Yk_k, Yk_k-1, ..., Yk_1)
        X_Entrada_i = np.hstack((X_conceroinicial[i], Yk_retrasos))
        X_Entrada_i = X_Entrada_i.reshape(1, -1)
        
        
        # Hacer la predicción
        prediccion = model.predict(X_Entrada_i).flatten()[0]
        # print(prediccion)
        
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
    prediction_flat = prediction_flat[:-1] # Quitar la ultima fila

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

# Entry point
if __name__ == '__main__':
    main()