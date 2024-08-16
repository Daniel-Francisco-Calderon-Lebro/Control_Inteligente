import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers import Dense

# Definir los datos de entrada y salida XOR
X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y = np.array([-1, 1, 1, -1])

# Crear el modelo
model = Sequential()
model.add(Dense(128, input_dim=2, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(4, activation='tanh'))
model.add(Dense(2, activation='tanh'))
model.add(Dense(1, activation='tanh'))

# Compilar el modelo
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

# 4. Entrenamos la red
model.fit(X, y, epochs=10000, verbose=2)

# 5. Evaluamos la red
scores = model.evaluate(X, y)

# 6. Mostramos los resultados
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# 7. Mostramos la red
print(model.predict(X))

# Guardar el modelo
model.save('red_xor.h5')


