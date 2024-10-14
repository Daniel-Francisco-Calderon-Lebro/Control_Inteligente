import matplotlib.pyplot as plt
import numpy as np
import math

##################### Inicializacion Funcion de transferencia SO ########################
k_Gs_SO = 2  # Ganancia
omega_n_SO = np.sqrt(2)/2  # Frecuencia natural (sqrt(Wn^2 = 1/2))
zeta_SO = (7 * 2)/(4*np.sqrt(2))  # Factor de amortiguamiento
delta_SO = 0.8  # Paso de tiempo (discretización)
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

u = 0.0  # Inicialización de la variable de entrada
uc = 0.0  # Inicialización de la variable de control
y = 0.0  # Valor inicial de la salida
##################### Terminacion Inicializacion Regulador Tecnologico ################################

# Inicialización de pesos (valores pequeños y aleatorios suelen funcionar mejor)

#cargamos los pesos iniciales desde un archivo csv
# w = np.loadtxt('wSO.csv', delimiter=',', skiprows=1)
# we11 = w[0]
# we21 = w[1]
# we31 = w[2]
# we12 = w[3]
# we22 = w[4]
# we32 = w[5]
# we13 = w[6]
# we23 = w[7]
# we33 = w[8]
# v1 = w[9]
# v2 = w[10]
# v3 = w[11]
# alfa = w[12]
# alfa1 = w[13]
# nn = w[14]

we11 = -9.708699555616312482e-01
we21 = -9.708699555616312482e-01
we31 = 1.060639064916551466e-01
we12 = -9.708699555616312482e-01
we22 = -9.708699555616312482e-01
we32 = 1.060639064916551466e-01
we13 = -9.708699555616312482e-01
we23 = -9.708699555616312482e-01
we33 = 1.060639064916551466e-01
v1 = -3.829066544911494230e+00
v2 = -3.829066544911494230e+00
v3 = -1.086481567129982118e+00
alfa = 1.002214499801117098e+00
alfa1 = 1.000000000000000056e-01
nn = 1.000000000000000000e+00


# we11 = we21 = we31 = we12 = we22 = we32 = we13 = we23 = we33 = 0.0
# v1 = v2 = -2.0
# v3 = 0.0
# alfa1 = 0.1 # parámetro de la rata de aprendizaje dinámica
# nn = 1 # parámetro de la rata de aprendizaje dinámica
# alfa = 0.2 # valor inicial de la rata de aprendizaje dinámica

while t <= 10000:
    if t >= 500:
        u = 40
    if t >= 1500:
        u = 80   
    if t >= 2500:
        u = 100
    if t >= 3500:
        u = 100
    if t >= 4500:
        u = 75 
    if t >= 5500:
        u = 25
    if t >= 7000:
        u = 0
    if t >= 8000:
        u = 0
    if t >= 9000:
        u = 0   
    if t >= 10000:
        u = 0
    # if t >= 3500:
    #     u = 100
    # if t >= 4500:
    #     u = 75 
    # if t >= 6000:
    #     u = 25
    # if t >= 8000:
    #     u = 0

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
    if uc > 1.0:
        uc = 1.0
    elif uc < 0:
        uc = 0.0
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

    alfa = alfa
    print(ey)

############################Respuesta del sistema ################################
    uc1 = uc
    # desnormalizar uc
    umax = 100
    umin = 0
    uc1 = uc1 * (umax - umin) + umin
    

    # Almacenar los valores para graficar
    uuc.append(uc1)  # Entrada ajustada
    uu.append(u1)  # Entrada sin ajuste
    z_SO_LS.append(y1)   # Salida del sistema
    tt.append(t)  # Tiempo
    # Incremento del tiempo
    t += 1

############################Terminacion Respuesta del sistema ################################


#############################Gráfica de la salida normalizada############################
# Gráfica de la salida normalizada
# plt.plot(tt, normz, color="red", label="Salida (y)")
# Grafica de la salida sin normalizar
plt.plot(tt, z_SO_LS, color="blue", label="Salida (y) Lazo Cerrado SO")
plt.plot(tt, uu, color="red", label="Entrada (u) SP")
plt.plot(tt, uuc, color="green", label="Escalones de Control (uc)")
plt.title("Respuesta del Sistema")
plt.xlabel("Tiempo")
plt.ylabel("Salida (y) Normalizada")
plt.legend()
plt.show()

# # Guardar la matriz con entrada y salida
# planta = [uuc, z_PO_LS]
# planta = np.transpose(planta)


# Crear el archivo CSV que guarda los pesos we y v
np.savetxt('wSO.csv', np.transpose([we11, we12, we13, we21, we22, we23, we31, we32, we33, v1, v2, v3, alfa, alfa1, nn]), delimiter=',', header='w11,w12,w13,w21,w22,w23,w31,w32,w33,v1,v2,v3', comments='')

print(we11, we12, we13, we21, we22, we23, we31, we32, we33)
print(v1, v2, v3)
print(alfa, alfa1, nn)
print(zeta_SO, omega_n_SO, delta_SO)