import matplotlib.pyplot as plt
import numpy as np

# Parámetros del sistema
u = 0  # Entrada del sistema
k_Gs_PO = 2  # Ganancia
tao_Gs_PO = 7  # Constante de tiempo
delta_Gs_PO = 1.5  # Paso de tiempo
t = 0  # Tiempo inicial

# Inicialización de listas para almacenar resultados
z_PO_LS = []  # Salida del sistema (y)
z_PO_LA = []  # Salida del sistema (y_open)
y = 0  # Valor inicial de la salida
y_open = 0
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



    # Ecuación de actualización de la salida del sistema en lazo abierto
    y_open = ((((u * k_Gs_PO) - y_open) * delta_Gs_PO) / tao_Gs_PO) + y_open
    
    u = u - y  # Aquí es donde se realiza el ajuste de la entrada
    # Ecuación de actualización de la salida del sistema en lazo cerrado
    y = ((((u * k_Gs_PO) - y) * delta_Gs_PO) / tao_Gs_PO) + y
    
    


    # Almacenar los valores para graficar
    uu.append(u)  # Entrada ajustada
    z_PO_LS.append(y)   # Salida del sistema
    z_PO_LA.append(y_open)
    tt.append(t)  # Tiempo
    t += 1  # Incremento del tiempo

# Normalización de las salidas y entradas para graficar
zmax = max(z_PO_LS)
zmin = min(z_PO_LS)
umax = max(uu)
umin = min(uu)

normu = []
normz = []
for i in range(len(z_PO_LS)):
    nz = (z_PO_LS[i] - zmin) / (zmax - zmin)  # Normalizar salida
    normz.append(nz)
    nu = (uu[i] - umin) / (umax - umin)  # Normalizar entrada
    normu.append(nu)

# Gráfica de la salida normalizada
# plt.plot(tt, normz, color="red", label="Salida (y)")
# Grafica de la salida sin normalizar
plt.plot(tt, z_PO_LS, color="blue", label="Salida (y) Lazo Abierto")
plt.plot(tt, z_PO_LA, color="red", label="Salida (y) en Lazo Cerrado")
plt.title("Respuesta del Sistema")
plt.xlabel("Tiempo")
plt.ylabel("Salida (y) Normalizada")
plt.legend()
plt.show()

# Guardar la matriz con entrada y salida
planta = [uu, z_PO_LS]
planta = np.transpose(planta)
