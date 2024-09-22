import matplotlib.pyplot as plt
from control import tf, step_response
import control as cl

# # Parámetros de la función de transferencia
# k = 1.6222
# wn = 0.238
# zeta = 0.8

# # Definición de la función de transferencia G(s) = k * wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
# numerator = [k * wn**2]
# denominator = [1, 2 * zeta * wn, wn**2]
# G = tf(numerator, denominator)

# # Respuesta al escalón
# t, y = step_response(G)

# # Graficar la respuesta
# plt.plot(t, y)
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud')
# plt.grid(True)
# plt.show()

K = 6
Ti = 8

k = 1.6222
wn = 0.238
zeta = 0.8

    # Numerador y denominador de la función de transferencia
num = [k * wn**2]
den = [1, 2*zeta*wn, wn**2]

    # Funciones de Transferencia
sys = cl.tf(num, den)   # FT Planta

num1 = [K]
den1 = [1,0]   
sys0 = cl.tf(K, [1,0])       # FT Ganacia Proporcional (Kc)
sys1 = cl.tf([1], [Ti,1])     # FT Ganacia Integral (1/Ti*S)

    
sysc = sys0 + sys1       # FT Controlador PI

sysf = cl.series(sysc,sys)