import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from joblib import dump
arrays=np.load("min_max_fcmeans.npz") #Cargo los minimos y maximos analizados en KMEANS_TRABAJO_1.py
ymax=arrays['arr_0']
ymin=arrays['arr_1']
umax=arrays['arr_2']
umin=arrays['arr_3']
emax=arrays['arr_4']
emin=arrays['arr_5']

#print (umax)

BDP= pd.ExcelFile(r'C:\Users\USUARIO\Desktop\Control inteligente I\parafallos.xlsx').parse(sheet_name='datosPrueba')
#CArgo el archivo de excel
BDP=BDP.values.T#organizo los datos en columnas

t=BDP[0]#Asigno columna a t
y=BDP[1]#Asigno columna a y
u=BDP[2]#Asigno columna a u
e=BDP[3]#Asigno columna a e
p=BDP[4]#Asigno columna a p

ny=[]
nu=[]
ne=[]
#normalizar informacion
for i in range(len(t)):
    normy= (y[i]-ymin)/(ymax-ymin)
    ny.append(normy)
    normu = (u[i] - umin) / (umax - umin)
    nu.append(normu)
    norme = (e[i] - emin) / (emax - emin)
    ne.append(norme)

infonorm=[ny, nu, ne]
print(len(infonorm))
infonorm1=np.transpose(infonorm)

#CODIGO LLAMADO DEL MODELO
from joblib import load
fcm = load('fcmeans_2E.joblib')

#con el clasificador entrenado le colocamos los datos.
print('INFO: Cargado Clasificador K-means previamente entrenado')

#plt.plot(fcm.predict(infonorm1))
#plt.show()


# Mostrar
plt.subplot(3, 1, 1)
plt.plot(y, color='red')
plt.plot(u, color='blue')
plt.plot(e, color='green')
plt.plot(p, color='orange')
plt.ylabel("DATOS DE PROCESO NORMALIZADOS")
# Establecer título de subimagen
plt.title("Proceso a Analizar")
# Cree un nuevo subgrafo, cuadrícula 2x1, número de secuencia 2
plt.subplot(3, 1, 3)
plt.plot(fcm.predict(infonorm1))
# Establecer título de subimagen
plt.title("Clasificador Entrenado")
plt.ylabel("CLASES ANALIZADAS")
plt.xlabel("TIEMPO")
# Establecer título
plt.suptitle("Proceso y Clasificador")
plt.subplot(3, 1, 2)
plt.plot(fcm.soft_predict(infonorm1))
# Mostrar
plt.show()
#termina mostrar
plt.show()



