import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import ExcelWriter


# Cargar Datos
df1 = pd.read_excel('C:/Users/Jhon Willian/Documents/Visual/Data_02.xlsx')

# Matriz 1171 x 2
rows=8000

# Vector de Salida (Out)
out_colum = 'yk'
yk = target_data = df1[out_colum].iloc[:rows].values

# Vector de Salida (Out)
out_colum = 't'
tt = target_data = df1[out_colum].iloc[:rows].values

# Vector de Entrada (IN)
input_Columns ="uk"
uk = df1[input_Columns].iloc[:rows].values
# Maximos y Minimos

zmax = max(yk)
zmin = min(yk)
umax = max(uk)
umin = min(uk)

# Vectores de Desnormalizacion
dnou = []
dnoy = [] 

# Max y Min
zmax = 139.539999999999 
zmin = 27.1099999999999
umax = 50
umin = 0 

# Desnormalizacion
for i in range(len(yk)):
    dy = ((zmax - zmin)*yk[i]) + zmin
    dnoy.append(dy)
    du = ((umax - umin)*uk[i]) + umin
    dnou.append(du)

# Guardar Datos
# Crear DataFrme
data = {'t': tt,'yk': dnoy, 'uk': dnou}
df = pd.DataFrame(data)

# Crear archivo .xlsx
excel_file = 'Data_03.xlsx'

# Guardar DataFrame en archivo .xlsx
df.to_excel(excel_file, index=False)

# Mensaje de confirmación
print(f"data saved'{excel_file}'")

# Grafico
plt.grid()
plt.title('Desnormalizacion')
plt.plot(tt[0:len(yk)],dnoy,'k-',linewidth=3)
plt.plot(tt[0:len(yk)],dnou,'r-',linewidth=3)
plt.legend(['Entrada','Salida'],loc = "upper right")
plt.ylabel(' Y(K)',fontsize=14)
plt.xlabel('Time (s)',fontsize=14)
print(umax,umin,zmax,zmin)
plt.show()
