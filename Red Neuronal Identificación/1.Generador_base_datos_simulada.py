import matplotlib.pyplot as plt
import numpy as np

u = 0
k = 1.2
tao =1.5
delta = 0.2
t = 0
z = []
y = 0
tt = []
uu = []

while t <= 2300:

    if t >= 5:
        u = 1
    if t >= 100:
        u = 2
    if t >= 230:
        u = 3
    if t >= 345:
        u = 4
    if t >= 460:
        u = 5
    if t >= 575:
        u = 4.5
    if t >= 705:
        u = 2
    if t >= 820:
        u = 0
    if t >= 935:
        u = 1
    if t >= 1150:
        u = 4
    if t >= 1265:
        u = 5
    if t >= 1380:
        u = 0
    if t >= 1510:
        u = 2
    if t >= 1625:
        u = 3
    if t >= 1740:
        u = 4
    if t >= 1855:
        u = 5
    if t >= 1970:
        u = 4.5
    if t >= 2085:
        u = 3.5
    if t >= 2100:
        u = 2
    if t >= 2215:
        u = 0

        # y =((((u*k)-y)*delta)/tao+y)
    y = ((((u * delta * k) / tao) + y) * (1 / (1 + delta / tao)))
    # print(y)
    uu.append(u)
    z.append(y)
    tt.append(t)
    t = t + 1
    print(t)

plt.plot(tt, z, color="blue")

plt.plot(tt, uu, color="red")
plt.show()

# Creo archivo .txt con los datos
filename='data.txt'
data = np.vstack((tt,uu,z)).T
top = 'Time (sec), Heater 1 (%), Temperature 1 (degC)'
np.savetxt(filename, data, delimiter=',', header=top, comments='')


'''
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

plt.plot(tt, z, color="red")

#plt.plot(tm2, yk, color="green")
#
# vector de salida
yk = np.delete(normz, (0, 1))
ykk = np.array([yk])

yk_2 = np.delete(normz, (139, 140))
yk_1 = np.delete(normz, (0, 140))
uk_2 = np.delete(normu, (139, 140))
uk_1 = np.delete(normu, (0, 140))

nn = np.array([uk_1, uk_2, yk_1, yk_2])
yl = np.transpose(nn)
tm = np.array([tt])
tm1 = np.transpose(tm)
tm2 = np.delete(tm1, (139, 140))

model = Sequential()
model.add(Dense(6, input_dim=4, activation='linear'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['binary_accuracy'])
model.fit(yl, yk, epochs=400, workers=4)

scores = model.evaluate(yl, yk)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
# print (model.predict(yl).round())

plt.plot(tm2, model.predict(yl), color="red")
plt.plot(tm2, yk, color="green")

plt.show()
##
##model_json = model.to_json()
##with open ("model.json","w") as json_file:
##     json_file.write(model_json)
##model.save_weights("model.h5")
##print("Modelo_guardado")
'''