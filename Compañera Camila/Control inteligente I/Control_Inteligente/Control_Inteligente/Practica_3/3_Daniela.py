
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  
import random

u = 0 #Variable de control
k = 1
tao = 9
delta = 3 #intervalos de la simulacion

t = 0
z = []
y = 0
tt = []
uu = []
nnu = []

xe2 = 0
xe3 = 0
eyy = [0,0]
alfa = 0.1
uc = 0


 # Valores aleatorios wji
we11 = [random.random()]
we12 = [random.random()]
we13 = [random.random()]

we21 = [random.random()]
we22 = [random.random()]
we23 = [random.random()]

we31 = [random.random()]
we32 = [random.random()]
we33 = [random.random()]


# Valores aleatorios vj
ve1 = [random.random()]
ve2 = [random.random()]
ve3 = [random.random()]
#we11 = [-0.06114155114728852]
#we12 = [-0.08385552631854909]
#we13 = [-0.10592063263779113]

#we21 = [-0.06114155114728852]
#we22 = [-0.08385552631854909]
#we23 = [-0.10592063263779113]

#we31 = [-0.06114155114728852]
#we32 = [-0.08385552631854909]
#we33 = [-0.10592063263779113]


# Valores aleatorios vj
#ve1 = [-1.485330635305826]
#ve2 = [-1.4882239767204326]
#ve3 = [-1.4872658354467212]

ascending = True  # Bandera para el modo ascendente

while t <= 38000:

    if ascending:
        if t >= 0:
            u = 0
        if t >= 2000:
            u = 20
        if t >= 4000:
            u = 50
        if t >= 6000:
            u = 100
        if t >= 8000:
            u = 100
        if t >= 10000:
            u = 100
        if t >= 12000:
            u = 100
        if t >= 14000:
            u = 50
        if t >= 16000:
            u = 0
            ascending = False  # Cambiar a modo descendente
    else:
        if t >= 18000:
            u = 0
        if t >= 20000:
            u = 0
        if t >= 22000:
            u = 50
        if t >= 24000:
            u = 100
        if t >= 28000:
            u = 100
        if t >= 30000:
            u = 50
        if t >= 32000:
            u = 50
        if t >= 34000:
            u = 5
        if t >= 36000:
            u = 0
        
            ascending = True  # Cambiar a modo ascendente
    
    # Normalizacion de u
    umax = 100
    umin = 0
    nu = (u - umin) / (umax - umin)
    #print(nu)

    y = ((((uc * k) - y) * delta) / tao) + y
    uu.append(nu)
    #uu.append(uc)
    z.append(y)
    tt.append(t)
    t = t + 1

    #Control Autoajustable 
    ey  = nu - y
    eyy.append(ey)

    xe1 = ey 
    xe2 = eyy[t-1]
    xe3 = eyy[t-2]
    #print(xe1,xe2,xe3)
    print(we11[t-1])
    
    he1 = (we11[t-1]*xe1)+(we21[t-1]*xe2)+(we31[t-1]*xe3)
    print(he1)
    he1 = 1/(1+np.exp(-he1))
    
    he2 = (we12[t-1]*xe1)+(we22[t-1]*xe2)+(we32[t-1]*xe3)
    he2 = 1/(1+np.exp(-he2))

    he3 = (we13[t-1]*xe1)+(we23[t-1]*xe2)+(we33[t-1]*xe3)
    he3 = 1/(1+np.exp(-he3))

    uc = (ve1[t-1]*he1)+(ve2[t-1]*he2)+(ve3[t-1]*he3)
    uc = 1/(1+np.exp(-uc))

    #Actualizacion de pesos 
    s = ey*uc*(1-uc)

    ve1a = ve1[t-2]+(alfa*s*he1)
    ve1.append(ve1a)
    ve2a = ve2[t-2]+(alfa*s*he2)
    ve2.append(ve2a)
    ve3a = ve3[t-2]+(alfa*s*he3)
    ve3.append(ve3a)

    s1 = s*ve1[t-2]*he1*(1-he1)
    s2 = s*ve1[t-2]*he2*(1-he2)
    s3 = s*ve1[t-2]*he3*(1-he3)

    we11a =  we11[t-2]+(alfa*s1*xe1)
    we11.append(we11a)
    we12a =  we12[t-2]+(alfa*s2*xe2)
    we12.append(we12a)
    we13a =  we13[t-2]+(alfa*s3*xe3)
    we13.append(we13a)

    we21a =  we21[t-2]+(alfa*s1*xe1)
    we21.append(we21a)
    we22a =  we22[t-2]+(alfa*s2*xe2)
    we22.append(we22a)
    we23a =  we23[t-2]+(alfa*s3*xe3)
    we23.append(we23a)

    we31a =  we31[t-2]+(alfa*s1*xe1)
    we31.append(we31a)
    we32a =  we32[t-2]+(alfa*s2*xe2)
    we32.append(we32a)
    we33a =  we33[t-2]+(alfa*s3*xe3)
    we33.append(we33a)

    #alfaa = 0.1 + (0.1 * np.abs(ey))
    alfaa= 0.5
    #alfaa = alfa[t-2] + (0.05 * np.abs(ey))
    alfa= alfaa

#print(len(z), len(tt), len(uu))

zmax = max(z)
zmin = min(z)
umax = max(uu)
umin = min(uu)

normu = []
normz = []
for i in range(len(z)):
    nz = (z[i] - zmin) / (zmax - zmin)
    normz.append(nz)
    nu = (uu[i] - umin) / (umax - umin)
    normu.append(nu)


# Crear un DataFrame de pandas con los datos
data = {'x': normu, 'y': normz}
df = pd.DataFrame(data)

# # Guardar el DataFrame en un archivo Excel
# excel_file = 'datos_planta.xlsx'
# df.to_excel(excel_file, index=False)

# Imprimir mensaje de confirmación
# print(f"Datos guardados en '{excel_file}'")
print(we11a,we12a,we13a,we21a,we22a,we23a,we31a,we32a,we33a,ve1a,ve2a,ve3a)


plt.plot(tt, z, color="red")
plt.plot(tt, uu, color="blue")

plt.show()
planta = [uu, z]
planta = np.transpose(planta)
