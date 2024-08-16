# Entrenamiento de una red XOR

# Importar librerías
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import joblib # para guardar y cargar el modelo

# Cargamos las 4 combinaciones de las compuertas XOR
training_data = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
target_data = np.array([[-1],[1],[1],[-1]])

# 1. Creamos la red
model = Sequential()
# 2. Agregamos las capas para evaluar la red de la xor

model.add(Dense(2, input_dim=2, activation='tanh'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='tanh'))

# model.add(Dense(16, input_dim=2, activation='linear'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='sigmoid'))
# model.add(Dense(1, activation='tanh'))

# 3. Compilamos la red
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

# 4. Entrenamos la red
model.fit(training_data, target_data, epochs=10000, verbose=2)

# 5. Evaluamos la red
scores = model.evaluate(training_data, target_data)

# 6. Mostramos los resultados
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# 7. Mostramos la red
print(model.predict(training_data))

# 8. Creamos el modelo .h5

model.save('XOR.h5')
# 8.1 guardamos el modelo con joblib
#joblib.dump(model, 'XOR.joblib')
