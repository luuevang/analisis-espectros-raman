import numpy as np
import matplotlib.pyplot as plt

datos = np.loadtxt('C:/Users/lucia/Dropbox/Labo 6 y 7/Raman/Prueba script analisis/AC3_1.txt')

xdata = datos[:,0]
ydata = datos[:,1]

fig = plt.figure(figsize = (4, 2))
plt.plot(xdata, ydata)
plt.legend(['Datos'], fontsize = 24)
plt.xlabel('Raman Shift [cm$^{-1}$]', fontsize = 24, ha = 'center')
plt.ylabel('Intensidad [u.a.]', fontsize = 24, ha = 'center')
plt.title('Espectro sin analizar', fontsize = 24, ha = 'center')
plt.show()