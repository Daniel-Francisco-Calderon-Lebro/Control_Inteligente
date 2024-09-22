import matplotlib.pyplot as plt
from tensorflow import keras
from keras import models
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import read_excel
 
##Llamamos la base de datos donde tenemos ya organizado los valores de entrada, salida, uk_1, uk_2, yk_1, yk_2, yk 
Base_Datos = 'data_1.xlsx'
df = pd.read_excel(Base_Datos, sheet_name='Hoja1')
#organizamos nuestros vectores de datos importados en arrays del script
rows=len(df.iloc[:, 0])
datos_salida =["yk"]
datos_entrada =["uk"]
De = df["uk"].iloc[:rows].values
Ds = df["yk"].iloc[:rows].values

zmax = max(Ds)
zmin = min(Ds)
umax = max(De)
umin = min(De)

normu = [0,0]
normz = [0,0]
t=[0, 0]
## Normalizacion de los datos de entrada y salida 
for i in range(len(Ds)-2):
    t.append(i)
    #nz = Ds[i]
    nz = (Ds[i] - zmin) / (zmax - zmin)
    normz.append(nz)
    #nu = De[i]
    nu = (De[i] - umin) / (umax - umin)
    normu.append(nu)

## Vector donde se guarda los valores de los minimos y maximos de entrada y salida 
norm_data=[umin,umax,zmin,zmax]
for w in range(len(normu)-6):
    norm_data.append(0)

## Organizacion de los vectores de la misma longitud para hacer el dataframe conlos valores de uk_1,uk_2,yk_1,yk_2,yk y los valores de minimos y maximo
yk_1 = normz[1:len(normz)-1]
yk   = normz[:len(normz)-2]
uk_1 = normu[1:len(normu)-1]
uk_2 = normu[2:len(normz)]
yk_2 = normz[2:len(normz)]
#yk_2 = normz[2:len(normz)-2]
#yk   = normz[:len(normz)-2]

plt.plot(t, Ds, color="red")
plt.plot(t,De, color="black")
plt.show()


print(len(uk_1),len(uk_2),len(yk_1),len(yk_2),len(yk),len(norm_data),len(Ds),len(De))
# Elaboracion del dataframe, hoja de excel con los valores de entrada, salida, uk_1, uk_2, yk_1, yk_2, yk y los valores de minimos y maximo

df = pd.DataFrame({'Entrada': normu[:-2],
                   'Salida':  normz[:-2],
                   'uk_1':    uk_1,
                   'uk_2':    uk_2,
                   'yk_1':    yk_1,
                   'yk_2':    yk_2,
                   'yk'  :    yk,   
                   'datos_normalizacion' : norm_data})

writer = ExcelWriter('DATA_2.xlsx')
df.to_excel(writer, 'Hoja1', index=False)
writer._save()
