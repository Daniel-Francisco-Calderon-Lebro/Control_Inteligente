import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import ExcelWriter


# Parametros de FT
#k = 1.035
#tao = 0.918
#delta = 0.1


# Parametros de FT
k = 2.035     # Ganancia
wn = 0.238    # Frecuencia
zita = 0.9  # Amortiguamiento
delta = 4    # Difrencial

# Variables Tiempo, Entrada, Salida
t = 0
u = 0
y = 0

# Vectores de Variables
yy = [0, 0]
uu = [0, 0]
tt = [0, 0]

# Funcion de Cammbio de Escalones
def Time (u,i):
    j = ([100,250]
        ,[320,400,500]
        ,[550,630,700,800]
        ,[870,930,990,1050,1110,1170])
    
    # Inicio Escalon
    if i == 5:
        u = 10
    if i in j[0]: #  0  > i <= 250:
        u = u+25
    if i in j[1]: # 251 > i <= 450
        u = u-20
    if i in j[2]: # 451 > i >= 700
        u = u+15
    if i in j[3]: # 701 > i >= 825
        u = u-12
    return u

# Simulacion 
for i in range(1170):
    u = Time(u,i)
    #y = ((((u * k) - y) * delta) / tao) + y
    y = (1 / (1 / (delta**2) + zita * wn / delta)) * (wn**2 * k * u - (wn**2 - 2 / delta**2) * yy[t-1] - (1 / delta**2 - zita * wn / delta) * yy[t-2])

    uu.append(u)
    yy.append(y)
    tt.append(t)
    t = t + 1

# Maximos y Minimos
ymax = max(yy)
ymin = min(yy)
umax = max(uu)
umin = min(uu)

# Vectores de Normalizacion
normu = []
normy = []

# Normalizacion
for i in range(len(yy)):
    ny = (yy[i] - ymin) / (ymax - ymin)
    normy.append(ny)
    nu = (uu[i] - umin) / (umax - umin)
    normu.append(nu)

# Guardar Datos
# Crear DataFrme
data = {'t': tt,'yk': normy, 'uk': normu}
df = pd.DataFrame(data)

# Crear archivo .xlsx
excel_file = 'data_1.xlsx'

# Guardar DataFrame en archivo .xlsx
df.to_excel(excel_file, 'Hoja1',  index=False)

# Mensaje de confirmación
print(f"data saved'{excel_file}'")

# Grafico
plt.grid()
plt.title('Datos de Simulacion')
plt.plot(tt,uu,'k-',linewidth=3)
plt.plot(tt,yy,'r-',linewidth=2)
plt.legend(['Entrada','Salida'],loc = "upper right")
plt.ylabel(' Y(K) - U(K)',fontsize=14)
plt.xlabel('Time (s)',fontsize=14)
print(umax,umin,ymax,ymin)
plt.show()
