
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  
import random
from keras.models import model_from_json

# cargar pesos
json_file = open('model_01.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model_01.h5")
print("Cargado modelo desde disco.")

# Listas para almacenar datos
num_elements = 5001
# Inicialización de variables del sistema
uck_1 = np.zeros(num_elements)
Uk_2 = np.zeros(num_elements)
Yk_1 = np.zeros(num_elements)
Yk_2 = np.zeros(num_elements)

t = 0
yy = [0, 0]
y = 0
tt = [0, 0]
uu = [0, 0]
nnu = [0,0]

xe2 = 0
xe3 = 0
eyy = [0,0]
alfa = [0.5]
uc = 0
y_prediction = 0

# # # Valores aleatorios wji
#we11 = [0, random.random()]
#we12 = [0, random.random()]
#we13 = [0, random.random()]

#we21 = [0, random.random()]
#we22 = [0, random.random()]
#we23 = [0, random.random()]

#we31 = [0, random.random()]
#we32 = [0, random.random()]
#we33 = [0, random.random()]

# # # Valores aleatorios vj
#ve1 = [0, random.random()]
#ve2 = [0, random.random()]
#ve3 = [0, random.random()]

# # Valores aleatorios wji
we11 = [4.825301078757931]
we12 = [4.373625889270133]
we13 = [4.2870578524851135]

we21 = [3.907922192350839]
we22 = [4.689014515728915 ]
we23 = [4.511999938241049]

we31 = [4.352649995659984]
we32 = [4.022487670401355]
we33 = [4.275733807477662]

# # Valores aleatorios vj
ve1 = [1.0881255390955378]
ve2 = [0.04687280771044229]
ve3 = [0.02176002773291716]

while t <= 5000:
    u = 100
    if t >= 1000:
        u = 17
    if t >= 2000:
        u = 84
    if t >= 3000:
        u = 0
    if t >= 4000:
        u = 67
    if t >= 5000:
        u = 50

    print(u)
    # Normalizacion de u
    umax = 100
    umin = 0
    nu = (u - umin) / (umax - umin)
    uu.append(nu)

    #Control Autoajustable 
    ey  = nu - y_prediction
    eyy.append(ey)

    xe1 = ey 
    xe2 = eyy[t-1]
    xe3 = eyy[t-2]
    
    he1 = (we11[t-1]*xe1)+(we21[t-1]*xe2)+(we31[t-1]*xe1)
    he1 = 1/(1+np.exp(-he1))
    
    he2 = (we12[t-1]*xe1)+(we22[t-1]*xe2)+(we32[t-1]*xe2)
    he2 = 1/(1+np.exp(-he2))

    he3 = (we13[t-1]*xe1)+(we23[t-1]*xe2)+(we33[t-1]*xe3)
    he3 = 1/(1+np.exp(-he3))

    uc = (ve1[t-1]*he1)+(ve2[t-1]*he2)+(ve3[t-1]*he3)
    uc = 1/(1+np.exp(-uc))

    #Red/Identificada
    if t > 1:
        Uk_2[t] = uck_1[t - 1]
        uck_1[t] = uc
        Yk_2[t] = Yk_1[t - 1]
        Yk_1[t] = yy[t-1]
    
    # Entradas para el modelo de red neuronal
    inputs = np.array([[uck_1[t], Uk_2[t], Yk_1[t], Yk_2[t]]])
    y_prediction = loaded_model.predict(inputs)
    y_prediction = float(y_prediction[0][0])
    yy.append(y_prediction)
    
    #Actualizacion de pesos 
    s = ey*uc*(1-uc)
 
    ve1a = ve1[t-1]+(alfa[t-1]*s*he1)
    ve1.append(ve1a)
    ve2a = ve2[t-1]+(alfa[t-1]*s*he2)
    ve2.append(ve2a)
    ve3a = ve3[t-1]+(alfa[t-1]*s*he3)
    ve3.append(ve3a)

    s1 = s*ve1[t-1]*he1*(1-he1)
    s2 = s*ve1[t-1]*he2*(1-he2)
    s3 = s*ve1[t-1]*he3*(1-he3)

    we11a =  we11[t-1]+(alfa[t-1]*s1*xe1)
    we11.append(we11a)
    we12a =  we12[t-1]+(alfa[t-1]*s2*xe2)
    we12.append(we12a)
    we13a =  we13[t-1]+(alfa[t-1]*s3*xe3)
    we13.append(we13a)

    we21a =  we21[t-1]+(alfa[t-1]*s1*xe1)
    we21.append(we21a)
    we22a =  we22[t-1]+(alfa[t-1]*s2*xe2)
    we22.append(we22a)
    we23a =  we23[t-1]+(alfa[t-1]*s3*xe3)
    we23.append(we23a)

    we31a =  we31[t-1]+(alfa[t-1]*s1*xe1)
    we31.append(we31a)
    we32a =  we32[t-1]+(alfa[t-1]*s2*xe2)
    we32.append(we32a)
    we33a =  we33[t-1]+(alfa[t-1]*s3*xe3)
    we33.append(we33a)

    alfaa = 5 + (1 * np.abs(ey))
    #alfaa = 6
    alfa.append(alfaa)

    tt.append(t)
    t = t + 1

zmax = max(yy)
zmin = min(yy)
umax = max(uu)
umin = min(uu)

normu = [0]
normz = [0]

for i in range(len(yy)-1):
    #print(z[i], zmin, zmax)
    nz = (yy[i] - zmin) / (zmax - zmin)
    normz.append(nz)
    nu = (uu[i] - umin) / (umax - umin)
    normu.append(nu)

print(we11a,we12a,we13a,we21a,we22a,we23a,we31a,we32a,we33a,ve1a,ve2a,ve3a)

# Crear el gráfico
tt = tt[:len(yy)]

plt.plot(tt, [x * 100 for x in yy], color="red", label="Sistema")
plt.plot(tt, [x * 100 for x in uu], color="blue", label="Escalones")

# Configurar el estilo del texto
font_style = {
    'family': 'sans-serif',  # Tipo de fuente (por ejemplo, 'serif', 'sans-serif', 'monospace')
    'weight': 'bold',   # Negrita
    'size': 12           # Tamaño de la fuente
}

# Agregar título y etiquetas de ejes con estilo personalizado
plt.title("Controlador RNA-Autoajustable", fontdict=font_style)
plt.xlabel("Tiempo", fontdict=font_style)
plt.ylabel("y(t)", fontdict=font_style)

# Configurar el tamaño de la fuente de la leyenda
plt.legend(fontsize=14)

# Agregar una cuadrícula
plt.grid(True)

# Mostrar el gráfico
plt.show()

#planta = [uu, z]
#planta = np.transpose(planta)
