import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
from pandas import ExcelWriter

# Parametros de FT
k = 1.035     # Ganancia
wn = 0.238    # Frecuencia
zita = 0.818  # Amortiguamiento
delta = 4     # Difrencial

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
alfa= 0.1

# Simulacion
while t <=  3000:

    if t >= 100:
        u = 20
    if t >= 500:
        u = 45
    if t >= 800:
        u = 80
    if t >=1400:
        u = 100
    if t >= 1800:
        u = 75
    if t >= 2200:
        u = 30
    if t >= 2500:
        u = 10
    if t >= 2800:
        u = 0
    
    # Normalizacion U
    umax = 100
    umin = 0
    df = (u - umin) / (umax - umin)
    
    # Calculo de y(k)
    # Ecuacion Sistema de Segundo Orden
    y = (1 / (1 / (delta**2) + zita * wn / delta)) * (wn**2 * k * UK - (wn**2 - 2 / delta**2) * yy[t-1] - (1 / delta**2 - zita * wn / delta) * yy[t-2])
    
    # Guardado para Graficar
    uu.append(df)
    yy.append(y)
    tt.append(t)
    
    t = t + 1 # Aumento Tiempo
    
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
    he1 = 1 / (1 + np.exp(- he1)) # Salida Neruona CO_1

    he2=(w21[t-1] * xe1) + (w22[t-1] * xe2) + (w23[t-1] * xe3)
    he2 = 1 / (1 + np.exp(- he2)) # Salida Neruona CO_2

    he3=(w31[t-1] * xe1) + (w32[t-1] * xe2) + (w33[t-1] * xe3)
    he3 = 1 / (1 + np.exp(- he3)) # Salida Neruona CO_3

    # Capa Salida  
    UK= (v1[t-1] * he1) + (v2[t-1] * he2) + (v3[t-1] * he3)
    UK= 1 / (1 + np.exp(-UK)) # Salida Neurona CS_Out
    
    # Calculo de ajustes para Vj
    s= e* UK* (1-UK)
    
    # Calculo coeficientes de ponderación Vj
    v_1 = v1[t-2] + (alfa * s * he1)
    v1.append(v_1)
    v_2 = v2[t-2] + (alfa * s * he2)
    v2.append(v_2)
    v_3 = v3[t-2] + (alfa * s * he3)
    v3.append(v_3)

    # Calculo de ajustes para Wji
    s1 = s * v1[t-2] * he1 * (1 - he1)
    s2 = s * v2[t-2] * he2 * (1 - he2)
    s3 = s * v3[t-2] * he3 * (1 - he3)

    # Calculo coeficientes de ponderación Wji
    w_11 = w11[t-2] + (alfa * s1 * xe1)
    w11.append(w_11)
    w_12 = w12[t-2] + (alfa * s2 * xe2)
    w12.append(w_12)
    w_13 = w13[t-2] + (alfa * s3 * xe3)
    w13.append(w_13)

    w_21 = w21[t-2] + (alfa * s1 * xe1)
    w21.append(w_21)
    w_22 = w22[t-2] + (alfa * s2 * xe2)
    w22.append(w_22)
    w_23 = w23[t-2] + (alfa * s3 * xe3)
    w23.append(w_23)

    w_31 = w31[t-2] + (alfa * s1 * xe1)
    w31.append(w_31)
    w_32 = w32[t-2] + (alfa * s2 * xe2)
    w32.append(w_32)
    w_33 = w33[t-2] + (alfa * s3 * xe3)
    w33.append(w_33)
    
    # Tasa de aprendizaje
    #alfaa = alfa + (0.05 * np.abs(e))
    alfaa = alfa + (0.5* np.abs(e))
    #alfaa= 10
    alfa= alfaa
    print(t)

# Vectores de Desnormalizacion
normu = []
normy = []

# Desnormalizacion
for i in range(len(yy)):
    dz = 100*yy[i]
    normy.append(dz)
    du = 100*uu[i]
    normu.append(du)

# Guardar Datos
# Crear DataFrame
data = {'u_1': normy, 'u': normu, 't': tt}
df = pd.DataFrame(data)
# Crear archivo .xlsx
excel_file = 'data.xlsx'
# Guardar DataFrame en archivo .xlsx
df.to_excel(excel_file, index=False)
print('Data Saved')
print(w_11,w_12,w_13,w_21,w_22,w_23,w_31,w_32,w_33,v_1,v_2,v_3)

# Grafica
plt.grid()
plt.title(f'Control Neuronal Autoajustable Alfa: {alfaa}')
plt.plot(tt,normu,'b-',linewidth=2)
plt.plot(tt,normy,'r-',linewidth=2)
plt.legend(['U','U_1'],loc = "upper right")
plt.ylabel('U(K)',fontsize=14)
plt.xlabel('Time (s)',fontsize=14)
plt.show()

#-0.055306112345390006 0.9220215399834877 0.8985615229020895 -0.17875910374596904 0.8574257157505152 1.0497798249634802 -0.46099200132521007 0.46627927562889276 1.1138442311334311 -3.671131734000952 -2.270146798790338 -2.21544797002803