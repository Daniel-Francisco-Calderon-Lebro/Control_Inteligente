#probar modelo

import numpy as np
from keras.models import load_model
from joblib import dump, load

from tensorflow.keras.utils import plot_model
import plotly.graph_objects as go

model = load_model('XOR.h5')
# model_joblib = load('XOR_de_save.joblib')

# X = np.array([[0,0],[0,1],[1,0],[1,1]])

# print(model.predict(X))
# # print(model_joblib.predict(X))

# 9. Graficamos la respuesta de la red en 3D usando Plotly

# Rango de valores para las variables de entrada (reduce la densidad de la cuadrícula)
x = np.linspace(-1.5, 1.5, 20)
y = np.linspace(-1.5, 1.5, 20)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)


# Predecir las respuestas en cada punto
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        input_data = np.array([[X[i, j], Y[i, j]]])
        Z[i, j] = model.predict(input_data)[0, 0]

# Crear el gráfico 3D con Plotly
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

# Añadir etiquetas y título
fig.update_layout(
    scene=dict(
        xaxis_title='X1',
        yaxis_title='X2',
        zaxis_title='Respuesta'
    ),
    title='Respuesta de la Red Neuronal en 3D'
)

# Mostrar el gráfico
fig.show()

# 10. Graficamos la red
plot_model(model, to_file='XOR.png', show_shapes=True)
