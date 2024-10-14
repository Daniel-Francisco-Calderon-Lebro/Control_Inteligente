import math

######################## Inicializacion Valores inciales Regulador Tecnologico MLP ################################
xe = [0.0, 0.0, 0.0, 0.0]  # Errores de entrada inicializados a cero
sp = 0  # Escalones de entrada del sistema
alfa1 = 1  # Tasa de aprendizaje dinámica (más baja para una convergencia más estable)
nn = 1  # Otro parámetro de la tasa de aprendizaje dinámica
alfa = 4  # Valor inicial de la tasa de aprendizaje (ajustable según el rendimiento)
uc = 0.0  # Inicialización de la variable de control
Sensor = 0.0  # Valor inicial de la salida
t = 0.0  # Inicialización de tiempo

we11 = we21 = we31 = we12 = we22 = we32 = we13 = we23 = we33 = 0.0
v1 = 0.5
v2 = 0.5
v3 = 0


############################### Inicio Funciones creadas ################################
# Normnalizacion de una variable
def normalizar(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)

# Desnormalizacion de una variable
def desnormalizar(x, xmin, xmax):
    return x * (xmax - xmin) + xmin
############################ Fin Funciones creadas ################################

############################# Inicio Programa ###########################
# Simulación hasta t = 180
while t <= True:

########################### Actualizacion de la Entrada del sistema ###########################
    #el Sp es el valor de referencia y es recibido mediante un topico
    sp = 0
    umax = 100
    umin = 0
    sp = normalizar(sp, umin, umax)  # Normalización de la entrada este es el sp
######################## Fin Actualizacion de la Entrada del sistema ###########################


############################ Cálculo del error ################################
    ey = sp - Sensor #cálculo del error actual
    xe[1] = ey  # Error actual e(t)          xe1 = xe[1]
    xe[2] = xe[1]  # Error anterior e(t-1)      xe2 = xe[2]
    xe[3] = xe[2]  # Error trasanterior e(t-2)  xe3 = xe[3]
############################ Fin Cálculo del error################################# 

    
    
################################ Cálculo de capa oculta#################################
    he1 = (we11 * xe[1]) + (we21 * xe[2]) + (we31 * xe[3])
    he1 = 1 / (1 + math.exp(-he1))
    he2 = (we12 * xe[1]) + (we22 * xe[2]) + (we32 * xe[3])
    he2 = 1 / (1 + math.exp(-he2))
    he3 = (we13 * xe[1]) + (we23 * xe[2]) + (we33 * xe[3])
    he3 = 1 / (1 + math.exp(-he3))
    # Cálculo de la salida
    uc = (v1 * he1) + (v2 * he2) + (v3 * he3)
    uc = 1 / (1 + math.exp(-uc*0.8+5))
    
################################ Desnormalizacion de la variable de control ################
    ucmax = 100
    ucmin = 0
    uc_Real = desnormalizar(uc, ucmin, ucmax)
############################# Fin Desnormalizacion de la variable de control ################
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
################################ Fin Cálculo de capa oculta##############################


########################### Ecuación de actualización de la salida del sistema en lazo cerrado
    #desnormalizar y
    ymax = 100
    ymin = 0
    y1 = desnormalizar(Sensor, ymin, ymax)
########################## Fin Ecuación de actualización de la salida del sistema en lazo cerrado



############################### Inicio Actualización de los pesos y Alfa ################################
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

################################ Fin Actualización de los pesos y Alfa ################################

    # Incremento del tiempo
    t += 1

############################ Fin  while ###########################
