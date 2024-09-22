import matplotlib.pyplot as plt
import numpy as np
import math
import random

########################Inicializacion Funcion de transferencia PO########################
# Parámetros del sistema
k_Gs_PO = 2  # Ganancia
tao_Gs_PO = 7  # Constante de tiempo
delta_Gs_PO = 1.5  # Paso de tiempo
t = 0  # Tiempo inicial

# Inicialización de listas para almacenar resultados
z_PO_LS = []  # Salida del sistema (y) vector graficacion
tt = []  # Tiempo
uu = []  # Entrada (u)
uuc = []  # Escalones de control
########################Terminacion Inicializacion Funcion de transferencia PO########################


######################## Inicializacion Regulador Tecnologico ################################
# y = [0.0, 0.0]  # Salida de la planta [0:actual, 1:anterior]
xe = [0.0, 0.0, 0.0, 0.0]  # Errores de entrada inicializados a cero
sp = 100  # Escalones de entrada del sistema
alfa1 = 1  # Tasa de aprendizaje dinámica (más baja para una convergencia más estable)
nn = 4  # Otro parámetro de la tasa de aprendizaje dinámica
alfa = 0.1  # Valor inicial de la tasa de aprendizaje (ajustable según el rendimiento)
u = 0.0  # Inicialización de la variable de entrada
uc = 0.0  # Inicialización de la variable de control
y = 0.0  # Valor inicial de la salida

# Inicialización de pesos (valores pequeños y aleatorios suelen funcionar mejor)
we11 = random.random()# -15.5681
we21 = random.random()#-15.30
we31 = random.random()#-10.56
we12 = random.random()#-15.49
we22 = random.random()#9.82
we32 = random.random()#9.85
we13 = random.random()#-15.49
we23 = random.random()#-10.522
we33 = random.random()#9.85
v1 = random.random()#-2.6
v2 = random.random()#-0.88
v3 = random.random()#-2.050

################################Terminacion Inicializacion Regulador Tecnologico################################



# Simulación hasta t = 180
while t <= 200:
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
    
    u = (u - umin) / (umax - umin) # Normalización de la entrada este es el sp
    


    # Ecuación de actualización de la salida del sistema en lazo cerrado
    y = ((((uc * k_Gs_PO) - y) * delta_Gs_PO) / tao_Gs_PO) + y
    

    t += 1  # Incremento del tiempo

    ey = u - y #cálculo del error actual

    xe[1] = ey  # Error actual e(t)          xe1 = xe[1]
    xe[2] = xe[1]  # Error anterior e(t-1)      xe2 = xe[2]
    xe[3] = xe[2]  # Error trasanterior e(t-2)  xe3 = xe[3]
    
    
    
    

    
    
    
    # Cálculo de capa oculta
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
    we12 = we12 + (alfa * s2 * xe[2])
    we13 = we13 + (alfa * s3 * xe[3])

    we21 = we21 + (alfa * s1 * xe[1])
    we22 = we22 + (alfa * s2 * xe[2])
    we23 = we23 + (alfa * s3 * xe[3])

    we31 = we31 + (alfa * s1 * xe[1])
    we32 = we32 + (alfa * s2 * xe[2])
    we33 = we33 + (alfa * s3 * xe[3])

    alfa = nn + (alfa1 * abs(ey))
    # alfa = 0.5

    # alfa = alfa
    print(alfa)

############################Respuesta del sistema ################################

    

    # Almacenar los valores para graficar
    uuc.append(uc)  # Entrada ajustada
    uu.append(u)  # Entrada sin ajuste
    z_PO_LS.append(y)   # Salida del sistema
    tt.append(t)  # Tiempo
    

############################Terminacion Respuesta del sistema ################################




############################Normalización de las salidas y entradas para graficar########################
# Normalización de las salidas y entradas para graficar
zmax = max(z_PO_LS)
zmin = min(z_PO_LS)
umax = max(uuc)
umin = min(uuc)

normu = []
normz = []
for i in range(len(z_PO_LS)):
    nz = (z_PO_LS[i] - zmin) / (zmax - zmin)  # Normalizar salida
    normz.append(nz)
    nu = (uuc[i] - umin) / (umax - umin)  # Normalizar entrada
    normu.append(nu)

###########################Terminacion Normalización de las salidas y entradas para graficar########################




#############################Gráfica de la salida normalizada############################
# Gráfica de la salida normalizada
# plt.plot(tt, normz, color="red", label="Salida (y)")
# Grafica de la salida sin normalizar
plt.plot(tt, z_PO_LS, color="blue", label="Salida (y) Lazo Cerrado PO")
plt.plot(tt, uu, color="red", label="Entrada (uc) ajustada")
# plt.plot(tt, uu, color="green", label="Entrada (u) sin ajuste")
plt.title("Respuesta del Sistema")
plt.xlabel("Tiempo")
plt.ylabel("Salida (y) Normalizada")
plt.legend()
plt.show()

# Guardar la matriz con entrada y salida
planta = [uuc, z_PO_LS]
planta = np.transpose(planta)

print(we11, we12, we13, we21, we22, we23, we31, we32, we33)
print(v1, v2, v3)

print(alfa)
