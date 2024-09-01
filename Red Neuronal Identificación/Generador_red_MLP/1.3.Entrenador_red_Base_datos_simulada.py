import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers import Dense
data = np.loadtxt(r'Red Neuronal Identificación\Generador_red_MLP\vectores_desplazados.txt', delimiter=',', skiprows=1)

# Función para separar X1, X2, y y hallar el numero de retrasos
k = (int(len(data[0])/2))
def seprarar_X1_X2_y(data,k):
    X1 = data[:, :k]
    X2 = data[:, k:2*k]
    y = data[:, 2*k:]
    X = np.concatenate((X1, X2), axis=1)
    return X,y

X,y = seprarar_X1_X2_y(data,k)
print(data.shape)

# ######################Modelo Funciona Perfecto###################
# Crear el modelo
model = Sequential()
model.add(Dense(2, input_dim=k*2, activation='relu'))
model.add(Dense(1,activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_squared_error'])  # métrica a MSE

##################################################################



###################################################################
# # Crear el modelo 0.04% ERROR
# model = Sequential()
# model.add(Dense(2, input_dim=k*2, activation='linear'))
# model.add(Dense(1))  # Sin activación para salida continua

# # Compilar el modelo con métricas adecuadas 0.08 de ERROR
# model.compile(loss='mean_squared_error',
#               optimizer='adam',
#               metrics=['mean_absolute_error'])  # métrica a MAE

# Modelo XOR#####################################################
# model = Sequential()
# model.add(Dense(64, input_dim=10, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))  # Sin activación para salida continua

############################################################################

# # Para este caso, funciona bien 0.26% ERROR
# model.compile(loss='mean_squared_error',
#               optimizer='adam',
#               metrics=['RootMeanSquaredError'])

# # Combinadas no da buena resultado
# model.compile(loss='mean_squared_error',
#               optimizer='adam',
#               metrics=['mean_absolute_error', 'RootMeanSquaredError'])


# # 11.59 de ERROR
# model.compile(loss='mean_squared_error',
#               optimizer='adam',
#               metrics=['binary_accuracy'])

#Nofunciona para este caso
# from tensorflow.keras.losses import CategoricalCrossentropy
# loss_function = CategoricalCrossentropy()
# model.compile(loss=loss_function,
#               optimizer='adam',
#              metrics=['accuracy'])

# 11.57 ERROR
# model.compile(loss='mean_squared_error',
#               optimizer='adam',
#              metrics=['mean_squared_error'])

# 7968.90% de Error
# model.compile(loss='mean_squared_error',
#               optimizer='adam',
#              metrics=['mean_absolute_percentage_error']) # MAPE


# 4. Entrenamos la red
model.fit(X, y, epochs=1000, verbose=2)

# 5. Evaluamos la red
scores = model.evaluate(X, y)

# 6. Mostramos los resultados
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# 7. Mostramos la red
print(model.predict(X))

# Guardar el modelo
model.save('Modelo_entrenado.h5')