import joblib
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from joblib import dump
arrays=np.load("min_max_kmeans.npz") #Cargo los minimos y maximos analizados en KMEANS_TRABAJO_1.py
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

'''plt.plot(t, y, color='red')
plt.plot(t, u, color='blue')
plt.plot(t, e, color='green')
plt.plot(t, p, color='orange')
plt.show()'''

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
kmeans1 = load('kmeans_a.joblib')

#con el clasificador entrenado le colocamos los datos.
print('INFO: Cargado Clasificador K-means previamente entrenado')

kmeans2 = kmeans1.predict(infonorm1) #comando para evaluar nuevos datos
#print (kmeans2)

#plt.plot(t, kmeans2, color='purple')
#plt.plot(t, u, color='blue')
#plt.plot(t, e, color='green')
#plt.show()

# Cree una nueva subimagen, cuadrícula 2x1, número de serie 1, el primer número es el número de filas, el segundo número es el número de columnas, que indica la disposición de las subimágenes, y el tercer número es el número de serie de las subimágenes
plt.subplot(2, 1, 1)
plt.plot(t, y, color='red')
plt.plot(t, u, color='blue')
plt.plot(t, e, color='green')
plt.plot(t, p, color='orange')

plt.ylabel("DATOS DE PROCESO NORMALIZADOS")
# Establecer título de subimagen
plt.title("Proceso Real")
# Cree un nuevo subgrafo, cuadrícula 2x1, número de secuencia 2
plt.subplot(2, 1, 2)
plt.plot(t, kmeans2, color='purple')
# Establecer título de subimagen
plt.title("Clasificador Entrenado")
plt.ylabel("CLASES ANALIZADAS")
plt.xlabel("TIEMPO")
# Establecer título
plt.suptitle("Proceso y Clasificador entrenado")

# Mostrar
plt.show()