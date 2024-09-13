import numpy as np
import matplotlib.pyplot as plt
#importo librería matlab
from control.matlab import *
import sympy as sp


s = sp.Symbol('s')
t = sp.Symbol('t')


wn = 1
k = 1.6222
eps = 0.1

# wn = 1
# k = 1.6222
# eps = 0.1

Gs = (wn**2*k)/(s**2+2*eps*wn*s+wn**2)


inv = sp.inverse_laplace_transform(Gs, s, t)


print(inv)

#grafico la transformadainversa

for i in range(100):
    print(inv.subs(t, i))


# Gs = tf((wn**2*k)/(s**2+2*eps*wn*s+wn**2))


# inv = ilaplce(Gs)  

# print(Gs)

#hallo la transformada inversa

