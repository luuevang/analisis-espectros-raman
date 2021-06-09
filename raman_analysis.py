import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
import glob
import os
import xlsxwriter
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'serif'

# FUNCIONES MODELO
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

lista_archivos = glob.glob(os.path.join(os.getcwd(), 'Prueba script analisis', "*.txt"))

ajuste_bwf = []   # Parámetros de ajuste de la BWF   
ajuste_lor = []   # Parámetros de ajuste de la lorentziana  
xmax_bwf = []     # Lista con las posiciones reales del máx del pico G  
ymax_bwf = []     # f(w_max) - BWF 
ymax_lor = []     # f(w_max) - Lor 
muestra = []      # Lista para guardar los nombres de los archivos para el excel
IdIg = []         # Cociente entre las intensidades máximas del pico D y G

analisis_fail = []  #Lista para guardar los archivos cuyo análisis no converja

with PdfPages('Prueba script analisis/Resumen_resultados.pdf') as pdf:
    for arx in lista_archivos:
        filename = arx.split("\\")
        archivo = filename[-1].split('.')
        muestra.append(archivo[0])
        
        datos = np.loadtxt(arx)
        xdata = datos[:,0]
        ydata = datos[:,1]

        try:
            par_bwf, cov_bwf, x_bwf, y_bwf, w_max, y_max = analisis_bwf(xdata, ydata, arx)
            p_lor, cov_lor, x_lor, y_lor, lor_ymax = analisis_lor(xdata, ydata, arx)
            print('{} -- ANALIZADO'.format(archivo[0]))

            fig = plt.figure(figsize=(10,6))
            ax = plt.axes()
            ax.yaxis.set_major_locator(MultipleLocator(50))
            ax.yaxis.set_minor_locator(MultipleLocator(25))
            ax.xaxis.set_major_locator(MultipleLocator(200))
            ax.xaxis.set_minor_locator(MultipleLocator(100))
            # plt.yticks(np.arange(0, 250, 25))
            plt.plot(xdata, ydata,'y')
            plt.plot(xdata, bwf(xdata, *par_bwf), '--g', linewidth=1)
            plt.plot(w_max, bwf(w_max, *par_bwf), '*k')
            plt.plot(xdata, lorentz(xdata, *p_lor), '--m', linewidth=1)
            plt.plot(xdata, bwf(xdata, *par_bwf) + lorentz(xdata, *p_lor), 'b', linewidth=1)
            plt.legend(['Datos', 'BWF - Modelo', 'Max BWF', 'LOR - Modelo', 'Ajuste'], loc = 2, fontsize = 14)
            plt.xlabel('Raman Shift [cm$^{-1}$]', fontsize = 16, ha = 'center')
            plt.ylabel('Intensidad [u.a.]', fontsize = 16, ha = 'center')
            plt.title('Muestra {}'.format(archivo[0]), fontsize = 20, ha = 'center', weight = 'bold')
            # plt.savefig('Prueba script analisis\{}.png'.format(archivo[0]))
            # pdf.savefig(fig)
            plt.show()
            # plt.close()

            newp_bwf = par_bwf.tolist()
            ajuste_bwf.append(newp_bwf)
            newp_lor = p_lor.tolist()
            ajuste_lor.append(newp_lor)
            xmax_bwf.append(w_max)
            ymax_bwf.append(y_max)
            ymax_lor.append(lor_ymax)
            IdIg.append(lor_ymax/y_max)

        except UnboundLocalError:
            analisis_fail.append(arx)
            print('{} agregado a analisis_fail'.format(arx))

    print('---Se analizaron todos los archivos---')
    # print('Parámetros de ajuste BWF:',ajuste_bwf)
    # print('Parámetros de ajuste LOR:',ajuste_lor)
    # print('Cociente I(D)/I(G):', IdIg)

    estructura = []
    for valor in range(len(IdIg)):
        if IdIg[valor] <= 0.1:
            estructura.append('sp3 > 0.2')
        else: 
            estructura.append('sp3 < 0.2')

    sp3_mayor = estructura.count('sp3 > 0.2')
    sp3_menor = estructura.count('sp3 < 0.2')
    torta = [sp3_mayor, sp3_menor]
    labels = ['sp$^3$ > 0.2', 'sp$^3$ < 0.2']
    colores = ['orange', 'magenta']
    graf_torta = plt.figure()
    plt.pie(torta, labels = labels, colors = colores)
    plt.legend(loc = 'lower right')
    plt.title('Fracción de sp$^3$ en la muestra')
    # plt.savefig('Prueba script analisis\Pie_chart.png')
    # pdf.savefig(graf_torta)
    # plt.show()
    plt.close()
    
excel = xlsxwriter.Workbook('Prueba script analisis\Parametros de ajuste.xlsx')
worksheet = excel.add_worksheet()
number_format = excel.add_format({'num_format' : '####0.##'})
columnas = ['Muestra', 'Tiempo','I0', 'w0', 'Q', 'g_bwf', 'bwf x_max*', 'bwf y_max*', '', 'A', 'x0', 'g_lor', 'lor y_max*', '','I(D)/I(G)', 'Tipo de estructura']
worksheet.write_row(0, 0, columnas)
worksheet.write_column(1, 0, muestra)
worksheet.write_column(1, 6, xmax_bwf, number_format)
worksheet.write_column(1, 7, ymax_bwf, number_format)
worksheet.write_column(1, 12, ymax_lor, number_format)
worksheet.write_column(1, 14, IdIg, number_format)
worksheet.write_column(1, 15, estructura)

for row_num, row_data in enumerate(ajuste_bwf):
    for col_num, col_data in enumerate(row_data):
        worksheet.write(row_num + 1, col_num + 2, col_data, number_format)

for row_num, row_data in enumerate(ajuste_lor):
    for col_num, col_data in enumerate(row_data):
        worksheet.write(row_num + 1, col_num + 9, col_data, number_format)

excel.close()

print('PROCESO TERMINADO')




