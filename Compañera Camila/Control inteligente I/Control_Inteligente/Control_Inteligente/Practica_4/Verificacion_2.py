import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import models
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import read_excel
import tensorflow as tf

# Variables Tiempo, Entrada, Salida
t = 0
u = 0
y = 0

# Vectores de Variables
yy = []
uu = []
tt = []

## Cargar Modelo Json
json_file = open('model_01.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)

# Cargar Pesos
loaded_model.load_weights("model_01.h5")
print("Modelo Cargado")

# Cargar Datos
while t <=  3000:

    if t >= 100:
        u = 20
    if t >= 500:
        u = 45
    if t >= 800:
        u = 80
    if t >=1200:
        u = 100
    if t >= 1800:
        u = 75
    if t >=2400:
        u = 40
    if t >= 2800:
        u = 0
    
    
    # Normalizacion U
    umax = 100
    umin = 0
    df = (u - umin) / (umax - umin)
    uu.append(df)
    tt.append(t)
    
    t = t + 1 # Aumento Tiempo

# Inicializacion de vectores para verificacion de la red
yk  =[]
uk_1=[0]
uk_2=[0]
yk_1=[0]
yk_2=[0]

aux_uk_1=0
aux_uk_2=0
aux_yk_1=0
aux_yk_2=0

## Verificacion de la red 
for i in range(1 ,len(uu),1):
    yk.append(float(loaded_model.predict([[aux_uk_1,aux_uk_2,aux_yk_1,aux_yk_2]])))
    aux_yk_1=float(yk[-1])
    yk_1.append(aux_yk_1)
    aux_yk_2=float(yk_1[-1])
    yk_2.append(aux_yk_2)
    aux_uk_1 = float(uu[i])
    aux_uk_2 = float(uu[i-1])
    print(i)

plt.grid()
plt.title('Verificacion Modelo Relu')
plt.plot(uu,'b-',linewidth=3)
plt.plot(yk,'r-',linewidth=2)
plt.legend(['U(k)','Y(K)'],loc = "upper right")
plt.ylabel(' Y(K) - U(k) ',fontsize=14)
plt.xlabel('Time (s)',fontsize=14)
plt.show()