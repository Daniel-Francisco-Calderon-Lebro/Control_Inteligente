import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
from pandas import ExcelWriter
from keras.models import model_from_json

# Cargar Modelo y Pesos
json_file = open('model_03.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Cargar Pesos
loaded_model.load_weights("model_03.h5")
print("Loaded Model")

# Vectores del Modelo
n = 3001
uk_1=np.zeros(n)
uk_2=np.zeros(n)
yk_1=np.zeros(n)
yk_2=np.zeros(n)

# Inicializacion Variables = Tiempo, Entrada, Salida, Error
t = 0
u = 0
y = 0
e = 0 

# Error
e_1 = [0,0]
xe2 = 0
xe3 = 0

# Ley de Control
UK = 0

# Vectores de Graficas
yy = [0, 0]
uu = [0, 0]
tt = [0, 0]

# Coeficientes de ponderación Wij
w11=[0.19282510546322223]
w12=[0.862739590870632]
w13=[0.1295868145233404]
w21=[0.538540639016772]
w22=[0.945355803552459]
w23=[0.2098691622333184]
w31=[0.5429110251883016]
w32=[0.6498954960850759]
w33=[-0.17507390775645532]

# Coeficientes de ponderación Vj
v1=[-2.0572266433504276]
v2=[-0.89763706667621]
v3=[-2.306772215120481]

# Taza de aprendizaje
alfa= 0.5

# Simulacion
while t <=  3000:

    if t >= 100:
        u = 2
    if t >= 500:
        u = 45
    if t >= 800:
        u = 80
    if t >=1400:
        u = 100
    if t >= 1800:
        u = 75
    if t >= 2000:
        u = 3
    if t >= 2500:
        u = 10
    if t >= 2800:
        u = 0
    
    # Normalizacion U
    umax = 100
    umin = 0
    df = (u - umin) / (umax - umin)

    # Control Autoajustable 

    # Calculpo del Error
    e  = df - y
    e_1.append(e)

    # Estados anterios del Error
    xe1 = e 
    xe2 = e_1[t-1]
    xe3 = e_1[t-2]
    
    #  Capa Oculta
    he1=(w11[t-1] * xe1) + (w21[t-1] * xe2) + (w31[t-1] * xe3)
    he1 = 1 / (1 + np.exp(- he1)) # Salida Neurona CO_1

    he2=(w21[t-1] * xe1) + (w22[t-1] * xe2) + (w23[t-1] * xe3)
    he2 = 1 / (1 + np.exp(- he2)) # Salida Neurona CO_2

    he3=(w31[t-1] * xe1) + (w32[t-1] * xe2) + (w33[t-1] * xe3)
    he3 = 1 / (1 + np.exp(- he3)) # Salida Neurona CO_3

    # Capa Salida  
    UK= (v1[t-1] * he1) + (v2[t-1] * he2) + (v3[t-1] * he3)
    UK= 1 / (1 + np.exp(-UK)) # Salida Neurona CS_Out
    
    # Valores del Modelo
    if t > 1:
        uk_1[t] = UK
        uk_2[t] = uk_1[t - 1]
        yk_1[t] = yy[t-1]
        yk_2[t] = yk_1[t - 1]
        
    
    # Entradas y para el modelo de red neuronal
    IN = np.array([[uk_1[t], uk_2[t], yk_1[t], uk_2[t]]])
    y = loaded_model.predict(IN)
    y = float(y[0][0])
   
    # Calculo de ajustes para Vj
    s = e* UK* (1-UK)
    
    # Calculo coeficientes de ponderación Vj
    v_1 = v1[t-1] + (alfa * s * he1)
    v1.append(v_1)
    v_2 = v2[t-1] + (alfa * s * he2)
    v2.append(v_2)
    v_3 = v3[t-1] + (alfa * s * he3)
    v3.append(v_3)

    # Calculo de ajustes para Wji
    s1 = s * v1[t-1] * he1 * (1 - he1)
    s2 = s * v2[t-1] * he2 * (1 - he2)
    s3 = s * v3[t-1] * he3 * (1 - he3)

    # Calculo coeficientes de ponderación Wji
    w_11 = w11[t-1] + (alfa * s1 * xe1)
    w11.append(w_11)
    w_12 = w12[t-1] + (alfa * s2 * xe2)
    w12.append(w_12)
    w_13 = w13[t-1] + (alfa * s3 * xe3)
    w13.append(w_13)

    w_21 = w21[t-1] + (alfa * s1 * xe1)
    w21.append(w_21)
    w_22 = w22[t-1] + (alfa * s2 * xe2)
    w22.append(w_22)
    w_23 = w23[t-1] + (alfa * s3 * xe3)
    w23.append(w_23)

    w_31 = w31[t-1] + (alfa * s1 * xe1)
    w31.append(w_31)
    w_32 = w32[t-1] + (alfa * s2 * xe2)
    w32.append(w_32)
    w_33 = w33[t-1] + (alfa * s3 * xe3)
    w33.append(w_33)
    
    # Tasa de aprendizaje
    #alfaa = alfa + ( 0.05* np.abs(e))
    alfaa = alfa + (0.5* np.abs(e))
    #alfaa = 10
    alfa = alfaa
    
    # Guardado Grafica
    yy.append(y)
    uu.append(df)
    tt.append(t)
    t = t + 1
    print(t)

# Crear DataFrame
data = {'u_1': yy, 'u': uu, 't': tt}
df = pd.DataFrame(data)
# Crear archivo .xlsx
excel_file = 'data.xlsx'
# Guardar DataFrame en archivo .xlsx
df.to_excel(excel_file, index=False)
print('Data Saved')
print(w_11,w_12,w_13,w_21,w_22,w_23,w_31,w_32,w_33,v_1,v_2,v_3)
# 0.19282510546322223 0.862739590870632 0.1295868145233404 0.538540639016772 0.945355803552459 0.2098691622333184 0.5429110251883016 0.6498954960850759 -0.17507390775645532 -2.0572266433504276 -0.89763706667621 -2.306772215120481

# Grafica
dnou = []
dnoy = [] 

# Desnormalizacion
for i in range(len(yy)):
    dy = 100*yy[i]
    dnoy.append(dy)
    du = 100*uu[i]
    dnou.append(du)

plt.grid()
plt.title(f'Control Neuronal Autoajustable Alfa: {alfaa:.2f}')
plt.plot(tt,dnou,'b-',linewidth=3)
plt.plot(tt,dnoy,'r-',linewidth=2)
plt.legend(['U(K)','Y(K)'],loc = "upper right")
plt.ylabel('U(K) - Y(K)',fontsize=14)
plt.xlabel('Time (s)',fontsize=14)
plt.show()

