import numpy as np

#Corregir la matriz
# G2
#matriz = np.array([
#    [1, 5.652, -2.373, 1.8258, -0.4529],
#    [1, 13.4121, -3.8053, 3.8798, -1.0744],
#    [1, 21.5151, -5.1494, 5.8726, -1.7230],
#    [1, 29.1044, -6.3468, 7.6788, -2.3316],  # Corregir la coma
#    [1, 35.872, -7.3820, 9.2563, -2.0873]
#])
#
#vector = np.array([0.1359, 0.0032, -0.0007, 0, 0])
# G1
matriz = np.array([
   [1, 4.723, -2.373, 1.8258, -0.4529],
   [1, 11.2076, -3.805, 3.8798, -1.0744],
   [1, 17.9724, -5.1494, 5.8726, -1.7229],
   [1, 24.3239, -6.3468, 7.6788, -2.3316],  # Corregir la coma
   [1, 29.9806, -7.3820, 9.2563, -2.0873]]
)
vector = np.array([0.1163,0.0026,-0.0006,0,0])
#
# Convertir la matriz a tipo float para asegurar la compatibilidad de tipos
matriz = matriz.astype(float)

# Multiplicar el vector por cada columna de la matriz y luego sumar
resultado = np.sum(matriz * vector[:, None], axis=0)

print("Matriz:")
print(matriz)
print("\nVector:")
print(vector)
print("\nResultado de la combinación lineal:")
print(resultado)