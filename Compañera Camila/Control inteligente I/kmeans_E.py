import matplotlib.pyplot as plt #creación de gráficos en dos dimensiones.
import pandas as pd  #Libreria para importar documentos de excel
import numpy as np #especializada en el cálculo numérico y el análisis de datos
from sklearn.cluster import KMeans #para hacer análisis predictivo
from joblib import dump #Para guardar el algoritmo entrenado

#PASO 1 importardatos de excel, extension .parse para ubicar la hoja que se requiere

datosprueba= pd.ExcelFile(r'C:\Users\USUARIO\Desktop\Control inteligente I\parafallos.xlsx').parse(sheet_name='datosentrenamiento')

BD=datosprueba.values.T #se almacena la base de datos cargada en la variable BD y transpuesta

#print(len(BD[1])) codigo para conocer la longitud de la columna 1 de la BD cargada
#print (datosprueba)

t=BD[0]#Tiempo columna 1
y=BD[1]#Exitacion Columna 2
u=BD[2]#Salida columna 3
e=BD[3]#Error columna 4
per=BD[4]#perturbacion columna 5

'''
plt.plot(t, y, color='red')
plt.plot(t, u, color='blue')
plt.plot(t, e, color='green')
plt.plot(t, per, color='black')
plt.show()''' #Funcion para presentar graficos delas variables antes cargadas

ymax=max(y)
ymin=min(y)
umax=max(u)
umin=min(u)
emax=max(e)
emin=min(e) #Funciones para determinar los maximos y minimos de cada variable para luego normalizar

ny=[] #inicio un vector ny vacio
nu=[] #inicio un vector nu vacio
ne=[] #inicio un vector ne vacio

#normalizar informacion
for i in range(len(t)):

    normy= (y[i]-ymin)/(ymax-ymin)
    ny.append(normy)# funcion para que cada que calcule el dato de normy sea almacenado en ny en la misma posicion

    normu = (u[i] - umin) / (umax - umin)
    nu.append(normu)# funcion para que cada que calcule el dato de normu sea almacenado en ny en la misma posicion

    norme = (e[i] - emin) / (emax - emin)
    ne.append(norme)# funcion para que cada que calcule el dato de norme sea almacenado en ny en la misma posicion

infonorm=[ny, nu, ne] #almaceno las variables normalizadas en infonorm
#print(len(infonorm))
infonorm1=np.transpose(infonorm)#transponer los datos para volverlos columnas
#print (infonorm1)

'''plt.plot(t, ny, color='red')
plt.plot(t, nu, color='blue')
plt.plot(t, ne, color='green')
plt.show()'''

clusters = input('Por favor ingrese el número de clusters para entrenar: ')#cantidad de clases
print('INFO: Entrenando y Prediciendo con Clasificador K-means...')
clusters = int(clusters)

kmeans1 = KMeans(n_clusters=clusters, init='k-means++', random_state=42).fit(infonorm1) #clasificador Kmeans

centroides = kmeans1.cluster_centers_ #Buscador de centros de datos

#print(centroides)# Centros de las clases analizadas

dump(kmeans1, 'kmeans_a.joblib')#guarda el clasificador entrenado

np.savez("min_max_kmeans", ymax, ymin, umax, umin, emax, emin) #Guarda los maximos y minimos en un archivo llamado min_max
kmeans2 = kmeans1.predict(infonorm1)

#plt.plot(t, kmeans2, color='red')
#plt.plot(t, u, color='blue')
#plt.plot(t, e, color='green')
#plt.show()

plt.subplot(2, 1, 1)
plt.plot(t, y, color='red')
plt.plot(t, u, color='blue')
plt.plot(t, e, color='green')
plt.plot(t, per, color='orange')
plt.ylabel("DATOS DE PROCESO NORMALIZADOS")
# Establecer título de subimagen
plt.title("Proceso a Analizar")
# Cree un nuevo subgrafo, cuadrícula 2x1, número de secuencia 2
plt.subplot(2, 1, 2)
plt.plot(t, kmeans2, color='purple')
# Establecer título de subimagen
plt.title("Clasificador Entrenado")
plt.ylabel("CLASES ANALIZADAS")
plt.xlabel("TIEMPO")
# Establecer título
plt.suptitle("Proceso y Clasificador")

# Mostrar
plt.show()