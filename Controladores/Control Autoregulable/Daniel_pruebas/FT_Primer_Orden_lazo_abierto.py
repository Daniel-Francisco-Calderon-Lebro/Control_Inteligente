import matplotlib.pyplot as plt
import numpy as np

# Parámetros del sistema
u = 0  # Entrada del sistema
k = 2  # Ganancia
tao = 7  # Constante de tiempo
delta = 2.5  # Paso de tiempo
t = 0  # Tiempo inicial

# Inicialización de listas para almacenar resultados
z = []  # Salida del sistema (y)
y = 0  # Valor inicial de la salida
tt = []  # Tiempo
uu = []  # Entrada (u)

# Simulación hasta t = 180
while t <= 180:
    # Definir el valor de la entrada u en función del tiempo t
    if t >= 20:
        u = 40
    if t >= 40:
        u = 80   
    if t >= 60:
        u = 100
    if t >= 80:
        u = 100
    if t >= 100:
        u = 75 
    if t >= 120:
        u = 25
    if t >= 160:
        u = 0
    umax = 100
    umin = 0
    u = (u - umin) / (umax - umin)  # Normalización de la entrada
    # Ecuación de actualización de la salida del sistema
    u = u - y
    y = ((((u * k) - y) * delta) / tao) + y
    
    # Ajuste de la entrada en función de la salida (lazo cerrado)
    

    # Almacenar los valores para graficar
    uu.append(u)  # Entrada ajustada
    z.append(y)   # Salida del sistema
    tt.append(t)  # Tiempo
    t += 1  # Incremento del tiempo

    # Mostrar la salida actual
    print(f"Tiempo: {t}, Salida: {y}")

# Normalización de las salidas y entradas para graficar
zmax = max(z)
zmin = min(z)
umax = max(uu)
umin = min(uu)

normu = []
normz = []
for i in range(len(z)):
    nz = (z[i] - zmin) / (zmax - zmin)  # Normalizar salida
    normz.append(nz)
    nu = (uu[i] - umin) / (umax - umin)  # Normalizar entrada
    normu.append(nu)

# Gráfica de la salida normalizada
plt.plot(tt, z, color="red", label="Salida (y)")
plt.title("Respuesta del sistema de Segundo Orden (Lazo Abierto)")
plt.xlabel("Tiempo")
plt.ylabel("Salida (y)")
plt.legend()
plt.show()

# Guardar la matriz con entrada y salida
planta = [uu, z]
planta = np.transpose(planta)
