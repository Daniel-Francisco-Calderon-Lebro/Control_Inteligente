import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers import Dense

data = np.loadtxt(r'Red Neuronal Identificación\vectores_desplazados.txt', delimiter=',', skiprows=1)

X1 = data[:, :5]
X2 = data[:, 5:10]
y = data[:, 10:]

X = np.concatenate((X1, X2), axis=1)

# Crear el modelo
model = Sequential()
model.add(Dense(64, input_dim=10, activation='linear'))
model.add(Dense(32, activation='linear'))
model.add(Dense(16, activation='linear'))
model.add(Dense(8, activation='linear'))
model.add(Dense(1))  # Sin activación para salida continua

# model = Sequential()
# model.add(Dense(64, input_dim=10, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))  # Sin activación para salida continua

# # Compilar el modelo
# model.compile(loss='mean_squared_error',
#               optimizer='adam',
#               metrics=['binary_accuracy'])

# Compilar el modelo con métricas adecuadas
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_absolute_error'])  # métrica a MAE



# 4. Entrenamos la red
model.fit(X, y, epochs=10000, verbose=2)

# 5. Evaluamos la red
scores = model.evaluate(X, y)

# 6. Mostramos los resultados
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# 7. Mostramos la red
print(model.predict(X))

# Guardar el modelo
model.save('Modelo_identificacion.h5')


