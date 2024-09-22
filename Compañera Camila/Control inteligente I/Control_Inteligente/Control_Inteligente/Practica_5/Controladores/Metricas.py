import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

def pid_closed_loop_metrics(K, Ti, Td, dt=1.0, time_end=1000, reference=20.0):
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
    steady_state_error = abs(reference - response[-1])
    overshoot = ((np.max(response) - reference) / reference) * 100
    settling_time = calculate_settling_time(time, response, reference)
    rise_time = calculate_rise_time(time, response, reference)
    steady_state_error_in_percent = steady_state_error / reference * 100

    # Ponderar las métricas de desempeño y penalizar según su magnitud
    performance_score = (
        (steady_state_error_in_percent / 100) * 0.1 +
        (overshoot / 100) * 0.1 +
        (1 - np.exp(-settling_time / time_end)) * 0.2 +
        (1 - np.exp(-rise_time / time_end)) * 0.2
    )

    # Transformar el rendimiento en un valor entre 0 y 1
    normalized_performance = 1 / (1 + performance_score)

    # Imprimir y devolver el valor de desempeño normalizado
    print(f"Valor de Desempeño Normalizado: {normalized_performance}")

    # Grafica
    plt.grid()
    plt.title('Respuesta: Lazo Abierto VS Control PID')
    plt.plot(time, reference_response, 'r-', linewidth=2)
    plt.plot(time, response, 'b-', linewidth=2)
    plt.legend(['Lazo Abierto', 'Control PID'], loc="lower right")
    plt.ylabel(' Y(K)', fontsize=14)
    plt.xlabel('Tiempo (s)', fontsize=14)
    plt.show()

    return normalized_performance

def calculate_settling_time(time, response, reference, percent_overshoot=5):
    overshoot_index = np.argmax(response)
    settling_time_index = np.argmax(response > (1 + percent_overshoot / 100) * reference)
    return time[settling_time_index]

def calculate_rise_time(time, response, reference):
    return time[np.argmax(response > 0.9 * reference)]

# Parámetros del controlador PID
K = 1   # Ganancia del controlador proporcional
Ti = 20  # Tiempo integral del controlador
Td = 0.01  # Tiempo derivativo del controlador

# Llamar a la función para obtener el valor de desempeño normalizado
normalized_performance = pid_closed_loop_metrics(K, Ti, Td)