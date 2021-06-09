import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

def bwf(w, *p):
    """ Breit-Wigner-Fano function for G-peak modelling
        -------------------------------------------
        PARAMETERS:
        a = I0: peak intensity
        b = w0: peak position
        c = Q: 1/Q BWF coupling coefficient
        d = g: full width at half maximum (FWHM)
        -------------------------------------------
    """
    a = p[0]    
    b = p[1]    
    c = p[2]    
    d = p[3]    

    return a*(1 + 2*(w - b)/(c*d))**2/(1 + (2*(w - b)/d)**2)

def lorentz(w, *p):
    """ Lorentzian function for D-peak modelling
        -------------------------------------------
        PARAMETERS:
        A: area under the curve
        b = w0: peak position
        d = g: full width at half maximum (FWHM)
        -------------------------------------------
    """
    A = p[0]    
    b = p[1]    
    d = p[2]    

    return (2*A/np.pi)*d/(4*(w - b)**2 + d**2)

def analisis_bwf(xdata, ydata, filename):
    """ 
    Esta función hace el análisis del pico G
        1. Selecciona los datos correspondientes a la zona del pico G
        2. Hace el ajuste no lineal de los datos usando la función curve_fit y el modelo BWF
        3. Calcula la posición real del pico, dada por:
            w_max = w0 + g/2Q
    ------------------------------------------------------------------------
    RETURN:
        par: parámetros ajustados
        cov
        x_bwf, y_bwf: datos x e y correspondientes a la zona del pico G
        w_max: posición real del máximo
        y_max: intensidad máxima
            y_max = bwf(w_max, *par)
    ------------------------------------------------------------------------
    """
    index1 = np.where(xdata>1400)[0][0]
    index2 = np.where(xdata>1750)[0][0]
    x_bwf = xdata[index1 : index2]
    y_bwf = ydata[index1 : index2]
    
    i0 = y_bwf[np.argmax(y_bwf)] 
    w0 = x_bwf[np.argmax(y_bwf)]
    ancho = x_bwf[int((np.argmin(y_bwf) + np.argmax(y_bwf))/2)] - x_bwf[0]
    factorQ = - ancho/i0
    p = np.array([i0, w0, factorQ, ancho])

    try:
        par, cov = curve_fit(bwf, x_bwf, y_bwf, p0 = p)
        w_max = par[1] + par[3]/(2*par[2])  # Posición real del x_max
        y_max = bwf(w_max, *par)
    except RuntimeError:
        print('Error: falló el ajuste BWF del archivo {}'.format(filename))
        pass
    
    return par, cov, x_bwf, y_bwf, w_max, y_max

def analisis_lor(xdata, ydata, filename):
    """" Esta función hace el análisis del pico D
        1. Selecciona los datos correspondientes a la zona del pico D
        2. Hace el ajuste no lineal de los datos usando la función curve_fit y el modelo lorentziano
    ------------------------------------------------------------------------
    RETURN:
        p_lor: parámetros ajustados
        cov_lor
        x_lor, y_lor: datos x e y correspondientes a la zona del pico G
        lor_ymax: intensidad máxima
            lor_ymax = lorentz(lor_xmax, *p_lor)
    ------------------------------------------------------------------------
    """
    y_aux = ydata - bwf(xdata, *par_bwf)
    index3 = np.where(xdata>1180)[0][0]
    index4 = np.where(xdata>1400)[0][0]
    x_lor = xdata[index3 : index4]
    y_lor = y_aux[index3 : index4]

    A = (np.pi/8)*(x_lor[-1] - x_lor[0])**2
    x0 = x_lor[np.argmax(y_lor)]
    g = x_lor[int((np.argmin(y_lor) + np.argmax(y_lor))/2)] - x_lor[int(np.argmax(y_lor)/2)]
    s = np.array([A, x0, g])

    try:
        p_lor, cov_lor = curve_fit(lorentz, x_lor, y_lor, p0 = s)
        lor_xmax = p_lor[1]
        lor_ymax = lorentz(lor_xmax, *p_lor)
    except RuntimeError:
        print('Error: falló el ajuste LOR del archivo {}'.format(filename))

    return p_lor, cov_lor, x_lor, y_lor, lor_ymax



par_bwf, cov_bwf, x_bwf, y_bwf, w_max, y_max = analisis_bwf(xdata, ydata, datos)
p_lor, cov_lor, x_lor, y_lor, lor_ymax = analisis_lor(xdata, ydata, datos)


figg = plt.figure(figsize=(4,2))
ax = plt.axes()
plt.plot(xdata, ydata,'y')
plt.plot(xdata, bwf(xdata, *par_bwf), '--g', linewidth=1)
plt.plot(w_max, bwf(w_max, *par_bwf), '*k')
plt.plot(xdata, lorentz(xdata, *p_lor), '--m', linewidth=1)
plt.plot(xdata, bwf(xdata, *par_bwf) + lorentz(xdata, *p_lor), 'b', linewidth=1)
plt.legend(['Datos', 'BWF - Modelo', 'Max BWF', 'LOR - Modelo', 'Ajuste'], loc = 2, fontsize = 24)
plt.xlabel('Raman Shift [cm$^{-1}$]', fontsize = 24, ha = 'center')
plt.ylabel('Intensidad [u.a.]', fontsize = 24, ha = 'center')
plt.title('Espectro analizado', fontsize = 24, ha = 'center', weight = 'bold')
# plt.savefig('Prueba script analisis\{}.png'.format(archivo[0]))
# pdf.savefig(fig)
plt.show()