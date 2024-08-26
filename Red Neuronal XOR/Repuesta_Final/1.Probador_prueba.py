import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import pandas as pd

# Cargar el modelo
model = load_model(r'Red Neuronal XOR\Repuesta_Final\red_xor.h5')

# Crear una cuadrícula de puntos para el gráfico 3D
x1 = np.linspace(-1.5, 1.5, 100)
x2 = np.linspace(-1.5, 1.5, 100)
X1, X2 = np.meshgrid(x1, x2)# Crear una cuadrícula de puntos para el gráfico 3D
Z = model.predict(np.c_[X1.ravel(), X2.ravel()])
Z = Z.reshape(X1.shape)
print(Z.shape)
# Configurar el gráfico 3D con Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie
ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')

# Configurar los límites de los ejes y aspecto cuadrado
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
ax.set_box_aspect([1, 1, 1])  # Asegura que todos los ejes tengan la misma escala

# Etiquetas de los ejes
ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')
ax.set_zlabel('Output')

# Mostrar el gráfico
# plt.show()

# Crear el gráfico 3D con Plotly
fig = go.Figure(data=[go.Surface(z=Z, x=X1, y=X2, colorscale='Viridis')])

# Añadir etiquetas y título
fig.update_layout(
    scene=dict(
        xaxis_title='X1',
        yaxis_title='X2',
        zaxis_title='Output'
    ),
    title='Respuesta de la Red Neuronal en 3D'
)

# Mostrar el gráfico interactivo
fig.show()

# obtengo el vector de valores ideales [-1,-1]=-1 , [-1,1]=1, [1,-1]=1, [1,1]=-1
Yideal = np.ones(len(X1))
for i in range(Yideal.shape[0]):
    if x1[i] > 0.1 and x2[i] > 0.1:
        Yideal[i] = 1
    elif x1[i] > 0.1 and x2[i] < -0.1:
        Yideal[i] = -1
    elif x1[i] < -0.1 and x2[i] > 0.1:
        Yideal[i] = 1
    elif x1[i] < -0.1 and x2[i] < -0.1:
        Yideal[i] = -1
    else:
        Yideal[i] = 0

# obtengo el error medio cuadrático

error = np.sum((Yideal - Z[:, 0]) ** 2) / len(Yideal)

print('Error medio cuadrático:', error)

#imprimimos los primeros 10 valores de los vectores de entrada y el valor esperado

print(x1.shape, x2.shape, Yideal.shape, Z.shape[0], Z.shape[1])


# creamos un csv con los datos y la respuesta ideal y la respuesta de la red neuronal

data = {'X1': x1, 'X2': x2, 'Yideal': Yideal, 'Y1red': Z[:, 0]}   

df = pd.DataFrame(data)

df.to_csv('data.csv', index=False)
