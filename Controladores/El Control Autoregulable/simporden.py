import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers.core import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import model_from_json
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt #creación de gráficos en dos dimensiones.
import math

k = 2
tao = 7
delta = 1.5
t = 0
z = []
y = 0
tt = []
uu = []
u_ca= []

# Variables del controlador
#y = [0.0, 0.0]  # Salida del la planta [0:actual, 1:anterior]
xe = [0.0, 0.0, 0.0, 0.0]  # primera capa con errores de entrada
sp = 100 # escalones de o entradas del sistema
alfa1 = 1 # parámetro de la rata de aprendizaje dinámica
nn = 1 # parámetro de la rata de aprendizaje dinámica
alfa = 4 # valor inicial de la rata de aprendizaje dinámica
u = 0.0 #incializacion de variable... puede no ser necesaria pues mas adelante se calcula
u_c= 0
# Inicialización de pesos
we11 = we21 = we31 = we12 = we22 = we32 = we13 = we23 = we33 = 0#np.random.rand()
v1 = v2 = 0.5
v3 = 0.0 #np.random.rand()
#v3 = 0.0
while t <= 14000:

    if t >= 500:
        u = 1
    if t >= 1500:
        u = 2
    if t >= 3000:
        u = 3
    if t >= 4500:
        u = 4
    if t >= 6000:
        u = 10
    if t >= 7500:
        u = 4.5
    if t >= 9000:
        u = 3.5
    if t >= 10500:
        u = 2
    if t >= 12000:
        u = 0
    u=u/10
    ey = (u) - (y) #cálculo del error actual
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

    # bsCálculo de la salida
    u1 = (v1 * he1) + (v2 * he2) + (v3 * he3)
    uc = (1 / (1 + math.exp(-u1)))
    #uc = ((uc - 0.5) * 2)*1
    # u = (math.exp(u1) - math.exp(-u1)) / (math.exp(u1) + math.exp(-u1)) #tanh(u)
    #u=sp/100
    '''
    if u > 1.0:
        u = 1.0
    elif u < 0:
        u = 0.0
        if ey>100:
            ey=100
        if ey<0:
            ey=0
    '''
    y = ((((uc * delta * k) / tao) + y) * (1 / (1 + delta / tao)))
    # Cálculo de S
    s = ey * uc * (1 - uc)  # u / (math.sinh(u1) * math.sinh(u1))
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

    uu.append(u)
    z.append(y)
    tt.append(t)
    t = t + 1
    u_ca.append(uc)
    print(t)

plt.plot(tt, z, color="red")
plt.plot(tt, uu, color="green")
plt.plot(tt, u_ca, color="yellow")
plt.show()
'''
zmax = max(z)
zmin = min(z)
umax = max(uu)
umin = min(uu)

normu = []
normz = []
for i in range(len(z)):
    nz = (z[i] - zmin) / (zmax - zmin)
    normz.append(nz)
    nu = (uu[i] - umin) / (umax - umin)
    normu.append(nu)

plt.plot(tt, z, color="red")

#plt.plot(tm2, yk, color="green")
#
# vector de salida
yk = np.delete(normz, (0, 1))
ykk = np.array([yk])

yk_2 = np.delete(normz, (139, 140))
yk_1 = np.delete(normz, (0, 140))
uk_2 = np.delete(normu, (139, 140))
uk_1 = np.delete(normu, (0, 140))

nn = np.array([uk_1, uk_2, yk_1, yk_2])
yl = np.transpose(nn)
tm = np.array([tt])
tm1 = np.transpose(tm)
tm2 = np.delete(tm1, (139, 140))

model = Sequential()
model.add(Dense(6, input_dim=4, activation='linear'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['binary_accuracy'])
model.fit(yl, yk, epochs=400, workers=4)

scores = model.evaluate(yl, yk)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
# print (model.predict(yl).round())

plt.plot(tm2, model.predict(yl), color="red")
plt.plot(tm2, yk, color="green")

plt.show()
##
##model_json = model.to_json()
##with open ("model.json","w") as json_file:
##     json_file.write(model_json)
##model.save_weights("model.h5")
##print("Modelo_guardado")
'''