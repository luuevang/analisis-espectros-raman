# Análisis de espectros Raman
Este es un código para analizar espectros Raman, según el modelo propuesto por Ferrari _et al._ en el paper _Interpretation of Raman spectra of disordered and amorphous carbon_. La idea fue automatizar el proceso lo más posible, de forma tal que lo único que hay que hacer es poner el path de la carpeta donde están los archivos y correr el programa. El programa devuelve:
- Parámetros de ajuste de ambos picos y un par de operaciones hechas con esos parámetros, en un Excel.
- Gráficos de cada espectro que se analizó (por separado).
- Un pdf con todos los gráficos y un gráfico de torta que indica cuantas muestras tuvieron 
contenido de sp$^3$ mayor a 0.2 y cuantas menor.

Por cómo es el análisis propuesto, no es posible automatizarlo al 100%. Este script me devuelve todos los datos que necesito para poder finalizar el análisis en forma manual. El pdf es más que nada para ver los resultados todos juntos de forma rápida. 

**Objetivo para el futuro cercano: agregar una visualización tipo dashboard.** 


