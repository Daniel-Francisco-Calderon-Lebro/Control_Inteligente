import matplotlib.pyplot as plt
import numpy as np
import math

# Inicialización de la función de transferencia de primer orden (PO)
k_Gs_PO = 2          # Ganancia
tao_Gs_PO = 7        # Constante de tiempo
delta_Gs_PO = 1.5    # Paso de tiempo
t = 0                # Tiempo inicial

# Listas para almacenar resultados
z_PO_LS = []  # Salida del sistema (para graficar)
tt = []       # Tiempo
uu = []       # Entrada (sin ajuste)
uuc = []      # Entrada ajustada (control)

# Inicialización del regulador tecnológico
xe = [0.0, 0.0, 0.0, 0.0]  # Errores de entrada inicializados a cero
sp = 100                    # Escalones de entrada del sistema
alfa1 = 1                   # Factor de aprendizaje dinámico
nn = 1                      # Parámetro adicional para tasa de aprendizaje
alfa = 4                    # Valor inicial de la tasa de aprendizaje
u = 0.0                     # Entrada inicial
uc = 0.0                    # Entrada de control inicial
y = 0.0                     # Salida inicial

# Carga de los pesos iniciales desde archivo CSV
w = np.loadtxt('w.csv', delimiter=',', skiprows=1)
we11, we12, we13 = w[0], w[1], w[2]
we21, we22, we23 = w[3], w[4], w[5]
we31, we32, we33 = w[6], w[7], w[8]
v1, v2, v3 = w[9], w[10], w[11]

# we11 =  -1.26
# we21 = -2.08
# we31 = -0.731
# we12 = -0.7247
# we22 = -1.74
# we32 = -1.3048
# we13 = -1.3986
# we23 = -1.9877
# we33 = -0.4436
# v1 = -3.4989
# v2 = -4.9677
# v3 = -3.0176

# Simulación hasta t = 20000
while t <= 20000:
    # Cambios de escalones en la entrada en diferentes tiempos
    if t >= 200: u = 40
    if t >= 4000: u = 80
    if t >= 6000: u = 100
    if t >= 10000: u = 75
    if t >= 12000: u = 25
    if t >= 16000: u = 0

    # Actualización de la salida del sistema
    u1 = u
    ymax, ymin = 100, 0
    y = ((((uc * k_Gs_PO) - y) * delta_Gs_PO) / tao_Gs_PO) + y
    y1 = y * (ymax - ymin) + ymin  # Desnormalización de la salida

    # Normalización de la entrada
    umax, umin = 100, 0
    u = (u - umin) / (umax - umin)  # Normalización de la entrada

    # Cálculo del error actual
    ey = u - y
    xe[1], xe[2], xe[3] = ey, xe[1], xe[2]  # Actualización de errores

    # Cálculo de la capa oculta
    he1 = 1 / (1 + math.exp(-(we11 * xe[1] + we21 * xe[2] + we31 * xe[3])))
    he2 = 1 / (1 + math.exp(-(we12 * xe[1] + we22 * xe[2] + we32 * xe[3])))
    he3 = 1 / (1 + math.exp(-(we13 * xe[1] + we23 * xe[2] + we33 * xe[3])))

    # Cálculo de la salida de control
    uc = 1 / (1 + math.exp(-(v1 * he1 + v2 * he2 + v3 * he3)))

    # Cálculo de S y ajuste de los pesos de salida V
    s = ey * uc * (1 - uc)
    v1 += alfa * s * he1
    v2 += alfa * s * he2
    v3 += alfa * s * he3

    # Ajuste de los pesos de entrada we
    s1, s2, s3 = s * v1 * he1 * (1 - he1), s * v2 * he2 * (1 - he2), s * v3 * he3 * (1 - he3)
    we11, we12, we13 = we11 + alfa * s1 * xe[1], we12 + alfa * s2 * xe[1], we13 + alfa * s3 * xe[1]
    we21, we22, we23 = we21 + alfa * s1 * xe[2], we22 + alfa * s2 * xe[2], we23 + alfa * s3 * xe[2]
    we31, we32, we33 = we31 + alfa * s1 * xe[3], we32 + alfa * s2 * xe[3], we33 + alfa * s3 * xe[3]

    # Ajuste de la tasa de aprendizaje
    alfa = nn + (alfa1 * abs(ey))

    # Almacenamiento de valores para graficar
    uuc.append(uc)      # Entrada ajustada
    uu.append(u1)       # Entrada original
    z_PO_LS.append(y1)  # Salida del sistema
    tt.append(t)        # Tiempo

    t += 1  # Incremento del tiempo

# Normalización de las salidas y entradas para graficar
zmax, zmin = max(z_PO_LS), min(z_PO_LS)
umax, umin = max(uuc), min(uuc)

normu = [(uuc[i] - umin) / (umax - umin) for i in range(len(uuc))]
normz = [(z_PO_LS[i] - zmin) / (zmax - zmin) for i in range(len(z_PO_LS))]

# Gráfica de la salida
plt.plot(tt, z_PO_LS, color="blue", label="Salida (y) Lazo Cerrado PO")
plt.plot(tt, uu, color="red", label="Entrada (uc) ajustada")
plt.title("Respuesta del Sistema")
plt.xlabel("Tiempo")
plt.ylabel("Salida (y)")
plt.legend()
plt.show()

# Guardar los pesos ajustados en un archivo CSV
np.savetxt('w.csv', np.transpose([we11, we12, we13, we21, we22, we23, we31, we32, we33, v1, v2, v3]),
           delimiter=',', header='w11,w12,w13,w21,w22,w23,w31,w32,w33,v1,v2,v3', comments='')

# Mostrar los pesos ajustados
print(we11, we12, we13, we21, we22, we23, we31, we32, we33)
print(v1, v2, v3)
