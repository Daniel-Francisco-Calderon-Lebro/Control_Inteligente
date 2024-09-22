import matplotlib.pyplot as plt
import numpy as np
import math

############################# Inicialización de la Función de Transferencia PO #############################

# Parámetros del sistema
k_Gs_PO = 2        # Ganancia
tao_Gs_PO = 7      # Constante de tiempo
delta_Gs_PO = 1.5  # Paso de tiempo
t = 0              # Tiempo inicial

# Inicialización de listas para almacenar resultados
z_PO_LS = []  # Salida del sistema (y) para graficación
tt = []       # Tiempo
uu = []       # Entrada (u)
uuc = []      # Escalones de control (uc)

#################################### Fin Inicialización Función de Transferencia PO ####################################


#################################### Inicialización del Regulador Tecnológico ####################################

# Inicialización de variables
xe = [0.0, 0.0, 0.0, 0.0]  # Errores de entrada
sp = 100  # Escalones o entradas del sistema
alfa1 = 0.1  # Parámetro de la rata de aprendizaje dinámica
nn = 0.1     # Parámetro adicional de la rata de aprendizaje dinámica
alfa = 10    # Valor inicial de la rata de aprendizaje dinámica
u = 0.0      # Inicialización de la variable de entrada (puede calcularse más adelante)
uc = 0.0     # Salida del controlador ajustada
y = 0        # Valor inicial de la salida

# Inicialización de pesos
we11 = we21 = we31 = we12 = we22 = we32 = we13 = we23 = we33 = 0.1  # Pesos de la capa oculta
v1 = v2 = v3 = 0.1  # Pesos de la salida

#################################### Fin Inicialización Regulador Tecnológico ####################################


#################################### Simulación del sistema ####################################
# Simulación hasta t = 200
while t <= 200:
    # Configuración de la entrada en función del tiempo
    if t >= 20: u = 40
    if t >= 40: u = 80
    if t >= 60: u = 100
    if t >= 100: u = 75
    if t >= 120: u = 25
    if t >= 160: u = 0

    # Normalización de la entrada
    umin, umax = 0, 100
    u = (u - umin) / (umax - umin)
    ey = u - y  # Cálculo del error actual
    
    # Actualización de los errores de entrada
    xe[3] = xe[2]  # Error trasanterior e(t-2)
    xe[2] = xe[1]  # Error anterior e(t-1)
    xe[1] = ey     # Error actual e(t)

    # Cálculo de la capa oculta
    he1 = 1 / (1 + math.exp(-(we11 * xe[1] + we21 * xe[2] + we31 * xe[3])))
    he2 = 1 / (1 + math.exp(-(we12 * xe[1] + we22 * xe[2] + we32 * xe[3])))
    he3 = 1 / (1 + math.exp(-(we13 * xe[1] + we23 * xe[2] + we33 * xe[3])))

    # Cálculo de la salida ajustada (uc)
    u1 = v1 * he1 + v2 * he2 + v3 * he3
    uc = 1 / (1 + math.exp(-u1))
    
    # Limitación de la salida ajustada
    uc = max(0.0, min(uc, 1.0))

    # Cálculo de S
    s = ey * uc * (1 - uc)
    s1, s2, s3 = s * v1 * he1 * (1 - he1), s * v2 * he2 * (1 - he2), s * v3 * he3 * (1 - he3)

    # Actualización de pesos de salida
    v1 += alfa * s * he1
    v2 += alfa * s * he2
    v3 += alfa * s * he3

    # Actualización de pesos de entrada
    we11 += alfa * s1 * xe[1]
    we12 += alfa * s2 * xe[1]
    we13 += alfa * s3 * xe[1]
    we21 += alfa * s1 * xe[2]
    we22 += alfa * s2 * xe[2]
    we23 += alfa * s3 * xe[2]
    we31 += alfa * s1 * xe[3]
    we32 += alfa * s2 * xe[3]
    we33 += alfa * s3 * xe[3]

    # Actualización de la rata de aprendizaje dinámica
    alfa = nn + alfa1 * abs(ey)

    #################################### Respuesta del sistema en lazo cerrado ####################################
    y = ((((uc * k_Gs_PO) - y) * delta_Gs_PO) / tao_Gs_PO) + y

    # Almacenar valores para graficar
    uuc.append(uc)  # Entrada ajustada
    uu.append(u)    # Entrada sin ajuste
    z_PO_LS.append(y)  # Salida del sistema
    tt.append(t)    # Tiempo
    t += 1          # Incremento del tiempo

#################################### Fin Simulación del sistema ####################################


#################################### Normalización para la graficación ####################################
# Normalización de las salidas y entradas para graficar
zmax, zmin = max(z_PO_LS), min(z_PO_LS)
umax, umin = max(uuc), min(uuc)

normu, normz = [], []
for i in range(len(z_PO_LS)):
    normz.append((z_PO_LS[i] - zmin) / (zmax - zmin))  # Normalizar salida
    normu.append((uuc[i] - umin) / (umax - umin))      # Normalizar entrada

#################################### Fin Normalización para graficar ####################################


#################################### Graficación de los resultados ####################################
plt.plot(tt, z_PO_LS, color="blue", label="Salida (y) Lazo Cerrado PO")
plt.plot(tt, uuc, color="red", label="Entrada (uc) ajustada")
plt.title("Respuesta del Sistema")
plt.xlabel("Tiempo")
plt.ylabel("Salida (y) Normalizada")
plt.legend()
plt.show()

# Guardar la matriz con entrada y salida
planta = np.transpose([uuc, z_PO_LS])

#################################### Fin del código ####################################
