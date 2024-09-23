import matplotlib.pyplot as plt
import numpy as np
import math

##################### Inicializacion Funcion de transferencia SO ########################
k_Gs_SO = 2  # Ganancia
omega_n_SO = np.sqrt(1/2)  # Frecuencia natural (sqrt(Wn^2 = 1/2))
zeta_SO = ((7 * np.sqrt(2)) / 2)/10  # Factor de amortiguamiento
delta_SO = 0.5  # Paso de tiempo (discretización)
t = 0  # Tiempo inicial
y_ant1 = 0  # Primer valor anterior (y_{i-1})
y_ant2 = 0  # Segundo valor anterior (y_{i-2})
A= (1/(delta_SO**(2)) + (zeta_SO*omega_n_SO)/delta_SO)
B= (omega_n_SO**2)*k_Gs_SO
C= omega_n_SO**2 - (2/(delta_SO**2))
D= (1/(delta_SO**(2))) - ((zeta_SO*omega_n_SO)/delta_SO)

# Inicialización de listas para almacenar resultados
z_SO_LS = []  # Salida del sistema (y)
tt = []  # Tiempo
uu = []  # Entrada (u)
uuc = []  # Escalones de control


######################## Inicializacion Regulador Tecnologico ################################
# y = [0.0, 0.0]  # Salida de la planta [0:actual, 1:anterior]
xe = [0.0, 0.0, 0.0, 0.0]  # Errores de entrada inicializados a cero
sp = 100  # Escalones de entrada del sistema
alfa1 = 1  # Tasa de aprendizaje dinámica (más baja para una convergencia más estable)
nn = 0.1  # Otro parámetro de la tasa de aprendizaje dinámica
alfa = 4  # Valor inicial de la tasa de aprendizaje (ajustable según el rendimiento)
u = 0.0  # Inicialización de la variable de entrada
uc = 0.0  # Inicialización de la variable de control
y = 0.0  # Valor inicial de la salida
##################### Terminacion Inicializacion Regulador Tecnologico ################################

# Inicialización de pesos (valores pequeños y aleatorios suelen funcionar mejor)

#cargamos los pesos iniciales desde un archivo csv
w = np.loadtxt('wSO.csv', delimiter=',', skiprows=1)
we11 = w[0]
we21 = w[1]
we31 = w[2]
we12 = w[3]
we22 = w[4]
we32 = w[5]
we13 = w[6]
we23 = w[7]
we33 = w[8]
v1 = w[9]
v2 = w[10]
v3 = w[11]

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

while t <= 20000:
    if t >= 200:
        u = 40
    if t >= 4000:
        u = 80   
    if t >= 6000:
        u = 100
    if t >= 8000:
        u = 100
    if t >= 10000:
        u = 75 
    if t >= 12000:
        u = 25
    if t >= 16000:
        u = 0

    u1 = u
    # Cálculo de la salida del sistema de segundo orden en lazo abierto
    y = (1/A)*(B*uc - C*y_ant1 - D*y_ant2)
    # Actualización de valores anteriores
    y_ant2 = y_ant1
    y_ant1 = y
    # Desnormalizar y
    ymax = 100
    ymin = 0
    y1 = y * (ymax - ymin) + ymin
    # normalizar u
    umax = 100
    umin = 0
    u = (u - umin) / (umax - umin) # Normalización de la entrada este es el sp
    
    # Cálculo del error
    ey = u - y #cálculo del error actual
    # Errores dezplazados
    xe[1] = ey  # Error actual e(t)          xe1 = xe[1]
    xe[2] = xe[1]  # Error anterior e(t-1)      xe2 = xe[2]
    xe[3] = xe[2]  # Error trasanterior e(t-2)  xe3 = xe[3]

    # Cálculo de la capa oculta
    he1 = (we11 * xe[1]) + (we21 * xe[2]) + (we31 * xe[3])
    he1 = 1 / (1 + math.exp(-he1))
    he2 = (we12 * xe[1]) + (we22 * xe[2]) + (we32 * xe[3])
    he2 = 1 / (1 + math.exp(-he2))
    he3 = (we13 * xe[1]) + (we23 * xe[2]) + (we33 * xe[3])
    he3 = 1 / (1 + math.exp(-he3))
    # Cálculo de la salida
    uc = (v1 * he1) + (v2 * he2) + (v3 * he3)
    uc = 1 / (1 + math.exp(-uc))


    # #u = ((u - 0.5) * 2)*1
    # # u = (math.exp(u1) - math.exp(-u1)) / (math.exp(u1) + math.exp(-u1)) #tanh(u)
    # #u=sp/100
    # if uc > 1.0:
    #     uc = 1.0
    # elif uc < 0:
    #     uc = 0.0
    # # if ey>100:
    # #     ey=100
    # # if ey<0:
    # #     ey=0

    # Cálculo de S
    s = ey * uc * (1 - uc)  # uc / (math.sinh(u1) * math.sinh(u1))
    # Actualización de pesos de salida V
    v1 = v1 + (alfa * s * he1)
    v2 = v2 + (alfa * s * he2)
    v3 = v3 + (alfa * s * he3)
    
    s1 = s * v1 * he1 * (1 - he1)
    s2 = s * v2 * he2 * (1 - he2)
    s3 = s * v3 * he3 * (1 - he3)


    # Actualización de pesos de entrada we
    we11 = we11 + (alfa * s1 * xe[1])
    we12 = we12 + (alfa * s2 * xe[1])
    we13 = we13 + (alfa * s3 * xe[1])

    we21 = we21 + (alfa * s1 * xe[2])
    we22 = we22 + (alfa * s2 * xe[2])
    we23 = we23 + (alfa * s3 * xe[2])

    we31 = we31 + (alfa * s1 * xe[3])
    we32 = we32 + (alfa * s2 * xe[3])
    we33 = we33 + (alfa * s3 * xe[3])

    alfa = nn + (alfa1 * abs(ey))
    # alfa = 0.5

    # alfa = alfa
    print(alfa)

############################Respuesta del sistema ################################

    

    # Almacenar los valores para graficar
    uuc.append(uc)  # Entrada ajustada
    uu.append(u1)  # Entrada sin ajuste
    z_SO_LS.append(y1)   # Salida del sistema
    tt.append(t)  # Tiempo
    # Incremento del tiempo
    t += 1

############################Terminacion Respuesta del sistema ################################




############################Normalización de las salidas y entradas para graficar########################
# Normalización de las salidas y entradas para graficar
zmax = max(z_SO_LS)
zmin = min(z_SO_LS)
umax = max(uuc)
umin = min(uuc)

#############################Gráfica de la salida normalizada############################
# Gráfica de la salida normalizada
# plt.plot(tt, normz, color="red", label="Salida (y)")
# Grafica de la salida sin normalizar
plt.plot(tt, z_SO_LS, color="blue", label="Salida (y) Lazo Cerrado SO")
plt.plot(tt, uu, color="red", label="Entrada (u) SP")
# plt.plot(tt, uu, color="green", label="Entrada (u) sin ajuste")
plt.title("Respuesta del Sistema")
plt.xlabel("Tiempo")
plt.ylabel("Salida (y) Normalizada")
plt.legend()
plt.show()

# # Guardar la matriz con entrada y salida
# planta = [uuc, z_PO_LS]
# planta = np.transpose(planta)


# Crear el archivo CSV que guarda los pesos we y v
np.savetxt('wSO.csv', np.transpose([we11, we12, we13, we21, we22, we23, we31, we32, we33, v1, v2, v3]), delimiter=',', header='w11,w12,w13,w21,w22,w23,w31,w32,w33,v1,v2,v3', comments='')

print(we11, we12, we13, we21, we22, we23, we31, we32, we33)
print(v1, v2, v3)
print(alfa)