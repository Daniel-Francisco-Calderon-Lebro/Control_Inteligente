import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

def pid_closed_loop_response(K, Ti, Td, dt=1.0, time_end=1000, reference=20.0):
   # Función de transferencia de la planta (primer orden estándar)
    plant = ctrl.TransferFunction([0.0025], [1, 0.075, 0.00125])

    # Función de transferencia del controlador PID
    pid = ctrl.TransferFunction([K * Td, K, K / Ti], [1, 0])

    # Sistema en lazo cerrado
    system = ctrl.feedback(pid * plant)

    # Respuesta a una entrada paso unitario en lazo cerrado (con controlador PID)
    time, response = ctrl.step_response(reference * system, T=np.arange(0, time_end, dt))

    # Respuesta a lazo Abierto
    _, reference_response = ctrl.step_response(reference * plant, T=np.arange(0, time_end, dt))

    # Calcular las métricas de desempeño
    error = reference - response
    squared_error = error**2
    absolute_error = np.abs(error)
    time_weighted_absolute_error = (time * absolute_error).sum()
    mean_squared_error = squared_error.mean()

    # Calcular el valor normalizado para cada métrica
    normalized_ise = 1 / (1 + np.sum(squared_error) * dt)
    normalized_iae = 1 / (1 + np.sum(absolute_error) * dt)
    normalized_itae = 1 / (1 + time_weighted_absolute_error)
    normalized_mse = 1 / (1 + mean_squared_error)

    # Calcular el valor total normalizado
    total_normalized_performance = (
        0.25 * normalized_ise +
        0.25 * normalized_iae +
        0.25 * normalized_itae +
        0.25 * normalized_mse
    )

    # Imprimir y devolver el valor total normalizado
    print(f"Valor Total Normalizado: {total_normalized_performance}")

    # Grafica
    plt.grid()
    plt.title('Respuesta: Lazo Abierto VS Control PID')
    plt.plot(time, reference_response, 'r-', linewidth=2)
    plt.plot(time, response, 'b-', linewidth=2)
    plt.legend(['Lazo Abierto', 'Control PID'], loc="lower right")
    plt.ylabel(' Y(K)', fontsize=14)
    plt.xlabel('Tiempo (s)', fontsize=14)
    plt.show()


    return total_normalized_performance

# Parámetros del controlador PID
K = 1   # Ganancia del controlador proporcional
Ti = 20  # Tiempo integral del controlador
Td = 0.01  # Tiempo derivativo del controlador

# Llamar a la función para obtener el valor total normalizado
total_normalized_performance = pid_closed_loop_response(K, Ti, Td)