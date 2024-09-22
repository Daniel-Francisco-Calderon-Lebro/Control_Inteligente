# Practica 1 : Entrenamiento Red Neuronal Simple

# Librerias 
import numpy as np
import keras 
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm 

# Matriz de entradas (In)
training_data = np.array([[-1,-1],[-1,1],[1,-1],[1,1]], "float32")

# Matriz de salidas (Out)
target_data = np.array([[-1],[1],[1],[-1]], "float32")

# Inicio arquitectura de red neuronal 
model = Sequential()

# Capas: Estructura: model.add(Dense(# Neuronas (capa), # Entradas, Funcion de activacion))

model.add(Dense(32, input_dim=2, activation='tanh'))   # Capa Oculta 1
model.add(Dense(32, activation='softsign'))            # Capa Oculta 2
model.add(Dense(1, activation='softsign'))             # Capa de Salida

# Argumentos para el Aprendizaje
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])
 
# Entrenamiento red neuronal
model.fit(training_data, target_data, epochs=2000)
 
# Evaluacion red neuronal
scores = model.evaluate(training_data, target_data)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print (model.predict(training_data).round())

# Vectores de Graficas  
xtotal=[]
ytotal=[]

# Vectores de combinacion
vecx=np.arange(-1.5, 1.5, 0.1)
vecy=np.arange(-1.5, 1.5, 0.1)

for x2 in range (30):
    yt=[]

    for x1 in range(30):
        vec=vecx[x1],vecy[x2]
        vec=np.array(vec)
        vec= vec[np.newaxis]
        xtotal.append(vec)
        yf = model.predict(vec)
        yt.append(float(np.array(yf)))
    ytotal.append(np.array(yt))
    print(x2+1)

# Graph 3D
vecx,vecy= np.meshgrid(vecx,vecy)
ytotal=np.array(ytotal)
fig= plt.figure(1)
ax = p3.Axes3D(fig)
ax = fig.add_subplot( 111 , projection = '3d')
ax.plot_surface(vecx, vecy, ytotal,cmap = cm.coolwarm,rstride=1, cstride=1)
ax.set_zlim(-1.01,1.01)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
