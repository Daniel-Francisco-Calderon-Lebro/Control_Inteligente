# Variables del controlador
import  math


#################################Inicializacion Regulador Tecnologico################################
# y = [0.0, 0.0]  # Salida del la planta [0:actual, 1:anterior]
xe = [0.0, 0.0, 0.0, 0.0]  # primera capa con errores de entrada
sp = 0 # escalones de o entradas del sistema
alfa1 = 0.1 # parámetro de la rata de aprendizaje dinámica
nn = 0.1 # parámetro de la rata de aprendizaje dinámica
alfa = 0.2 # valor inicial de la rata de aprendizaje dinámica
u = 0.0 #incializacion de variable... puede no ser necesaria pues mas adelante se calcula

# Inicialización de pesos
we11 = we21 = we31 = we12 = we22 = we32 = we13 = we23 = we33 = 0.1
v1 = v2 = v3 = 0.1
while True: #este while se debe condicionar a las muestas que se le estén programando a la simulacion



    ey = 10 - 1 #cálculo del error actual
    xe[3] = xe[2]  # Error trasanterior e(t-2)  xe3 = xe[3]
    xe[2] = xe[1]  # Error anterior e(t-1)      xe2 = xe[2]
    xe[1] = ey  # Error actual e(t)          xe1 = xe[1]

    # Cálculo de capa oculta
    he1 = (we11 * xe[1]) + (we21 * xe[2]) + (we31 * xe[3])
    he1 = 1 / (1 + math.exp(-he1))
    he2 = (we12 * xe[1]) + (we22 * xe[2]) + (we32 * xe[3])
    he2 = 1 / (1 + math.exp(-he2))
    he3 = (we13 * xe[1]) + (we23 * xe[2]) + (we33 * xe[3])
    he3 = 1 / (1 + math.exp(-he3))

    # Cálculo de la salida
    u1 = (v1 * he1) + (v2 * he2) + (v3 * he3)
    u = 1 / (1 + math.exp(-u1 * 1.0))
    #u = ((u - 0.5) * 2)*1
    # u = (math.exp(u1) - math.exp(-u1)) / (math.exp(u1) + math.exp(-u1)) #tanh(u)
    #u=sp/100
    if u > 1.0:
        u = 1.0
    elif u < 0:
        u = 0.0
        '''if ey>100:
            ey=100
        if ey<0:
            ey=0
        '''
    # Cálculo de S
    s = ey * u * (1 - u)  # u / (math.sinh(u1) * math.sinh(u1))
    s1 = s * v1 * he1 * (1 - he1)
    s2 = s * v2 * he2 * (1 - he2)
    s3 = s * v3 * he3 * (1 - he3)

    # Actualización de pesos de salida V
    v1 = v1 + (alfa * s * he1)
    v2 = v2 + (alfa * s * he2)
    v3 = v3 + (alfa * s * he3)

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
