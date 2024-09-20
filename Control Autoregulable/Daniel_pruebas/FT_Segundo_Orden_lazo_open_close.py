import matplotlib.pyplot as plt
import numpy as np

# Parámetros del sistema de segundo orden
k = 2  # Ganancia
omega_n = np.sqrt(1/2)  # Frecuencia natural (sqrt(Wn^2 = 1/2))
zeta = ((7 * np.sqrt(2)) / 2)/10  # Factor de amortiguamiento
delta = 0.5  # Paso de tiempo (discretización)
t = 0  # Tiempo inicial

# Inicialización de listas para almacenar resultados
u = 0  # Entrada inicial

z = []  # Salida del sistema (y)
z_open = []  # Salida del sistema (y_open)

y = 0  # Valor inicial de la salida
y_open = 0

y_ant1 = 0  # Primer valor anterior (y_{i-1})
y_ant2 = 0  # Segundo valor anterior (y_{i-2})

y_ant1_open = 0 # Primer valor anterior de la salida en lazo abierto (y_{i-1})
y_ant2_open = 0 # Segundo valor anterior de la salida en lazo abierto (y_{i-2})
tt = []  # Tiempo
uu = []  # Entrada (u)

A= (1/(delta**(2)) + (zeta*omega_n)/delta)
B= (omega_n**2)*k
C= omega_n**2 - (2/(delta**2))
D= (1/(delta**(2))) - ((zeta*omega_n)/delta)

# Simulación hasta t = 180
while t <= 150:
    # Definir el valor de la entrada u en función del tiempo t
    if t >= 20:
        u = 40
    # if t >= 40:
    #     u = 80   
    # if t >= 60:
    #     u = 100
    # if t >= 80:
    #     u = 100
    # if t >= 100:
    #     u = 75 
    # if t >= 120:
    #     u = 25
    # if t >= 160:
    #     u = 0
    
    # Cálculo de la salida del sistema de segundo orden en lazo abierto
    y_open = (1/A)*(B*u - C*y_ant1_open - D*y_ant2_open)
    y_ant2_open = y_ant1_open
    y_ant1_open = y


    # Cálculo de la salida del sistema de segundo orden en lazo cerrado
    u = u - y # Aquí es donde se realiza el ajuste de la entrada
    y = (1/A)*(B*u - C*y_ant1 - D*y_ant2)

    # Actualización de valores anteriores
    y_ant2 = y_ant1
    y_ant1 = y
    

    # Almacenar los valores para graficar
    uu.append(u)  # Entrada ajustada
    z.append(y)   # Salida del sistema
    z_open.append(y_open)
    tt.append(t)  # Tiempo
    t += 1  # Incremento del tiempo

    # Mostrar la salida actual
    print(f"Tiempo: {t}, Salida: {y}")

# Gráfica de la salida
plt.plot(tt, z, color="blue", label="Salida (y) Lazo Cerrado")
plt.plot(tt, z_open, color="red", label="Salida (y) en Lazo Abierto")
plt.title("Respuesta del sistema de Segundo Orden")
plt.xlabel("Tiempo")
plt.ylabel("Salida (y)")
plt.legend()
plt.grid(True)
plt.show()
