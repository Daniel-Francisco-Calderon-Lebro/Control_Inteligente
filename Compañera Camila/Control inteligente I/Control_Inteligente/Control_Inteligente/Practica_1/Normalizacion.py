import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import ExcelWriter


# Cargar Datos
df1 = pd.read_excel('C:/Users/Jhon Willian/Documents/Visual/Practica_4/Datos/Libro3.xlsx')

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

# Vectores de Normalizacion
normu = []
normy = [] 

# Normalizacion
for i in range(len(yk)):
    ny = (yk[i] - zmin) / (zmax - zmin)
    normy.append(ny)
    nu = (uk[i] - umin) / (umax - umin)
    normu.append(nu)

# Guardar Datos
# Crear DataFrme
data = {'t': tt,'yk': normy, 'uk': normu}
df = pd.DataFrame(data)

# Crear archivo .xlsx
excel_file = 'Data_02.xlsx'

# Guardar DataFrame en archivo .xlsx
df.to_excel(excel_file, index=False)

# Mensaje de confirmación
print(f"data saved'{excel_file}'")

# Grafico
plt.grid()
plt.title('Normalizacion')
plt.plot(tt[0:len(yk)],normy,'k-',linewidth=3)
plt.plot(tt[0:len(yk)],normu,'r-',linewidth=3)
plt.legend(['Entrada','Salida'],loc = "upper right")
plt.ylabel(' Y(K)',fontsize=14)
plt.xlabel('Time (s)',fontsize=14)
print(umax,umin,zmax,zmin)
plt.show()
