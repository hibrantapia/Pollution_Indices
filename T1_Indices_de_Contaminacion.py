# COPIA Y PEGA CADA CHUNK DE CÓDIGO EN UNA CELDA DE COLAB, Y EJECUTA CADA CELDA.

#-----------------------------------------------------------------------#

!pip install numpy
!pip install pandas
!pip install unidecode
!pip install matplotlib

#-----------------------------------------------------------------------#

import unidecode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#-----------------------------------------------------------------------#

filename_2019 = '/content/indice_2019.csv'
data_2019 = pd.read_csv(filename_2019, encoding = 'latin-1', parse_dates = ['Fecha']) 

filename_2020 = '/content/indice_2020.csv'
data_2020 = pd.read_csv(filename_2020, encoding = 'latin-1', parse_dates = ['Fecha'])

filename_2021 = '/content/indice_2021.csv'
data_2021 = pd.read_csv(filename_2021, encoding = 'latin-1', parse_dates = ['Fecha'])

filename_2022 = '/content/indice_2022.csv'
data_2022 = pd.read_csv(filename_2022, encoding = 'latin-1', parse_dates = ['Fecha'])

#-----------------------------------------------------------------------#

combined_data = pd.concat([data_2019, data_2020, data_2021, data_2022], ignore_index=True) # Juntamos las cuatro bases de datos, en un solo dataframe.
combined_data = combined_data.sort_values(by = 'Fecha') # Ordena las fechas en orden cronológico.
combined_data.columns = [unidecode.unidecode(col) for col in combined_data.columns] # Normaliza los nombres de las columnas eliminando los acentos.

combined_data.head()
#-----------------------------------------------------------------------#

print(combined_data.shape)

#-----------------------------------------------------------------------#

combined_data_filled = combined_data.fillna("NaN")
combined_data_filled.head()

#-----------------------------------------------------------------------#

#NORMAL

regions = ['Noroeste', 'Noreste', 'Centro', 'Suroeste', 'Sureste']
gases = ['ozono', 'dioxido de azufre', 'dioxido de nitrogeno', 'monoxido de carbono', 'PM10', 'PM25']
colors = ['red', 'magenta', 'lime', 'teal', 'navy']

fig, axs = plt.subplots(3, 2, figsize=(20,10))  # 3 filas, 2 columnas.

# Loop de cada gas.
for gas_index, gas in enumerate(gases):
    row = gas_index // 2
    col = gas_index % 2
    
    # Loop de cada región.
    for i, region in enumerate(regions):
        # Crea un nombre de columna combinando la región y el gas.
        column_name = f'{region} {gas}'
        
        # Plot de los datos para dicha región y gas, con etiqueta y color.
        axs[row, col].plot(combined_data['Fecha'], combined_data[column_name], color = colors[i], label = region)
    
    axs[row, col].set_title(f'Índice de {gas} en las 5 regiones de la Ciudad de México (2019 - 2022)')
    axs[row, col].set_xlabel('Fecha')
    axs[row, col].set_ylabel('Índice')
    axs[row, col].legend(loc='upper right')

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------#

# Interpola los datos de cada columna
for col in combined_data.columns:
    if col != 'Fecha':  # Todas las columnas menos la de Fecha
        combined_data[col] = combined_data[col].interpolate()
        
#-----------------------------------------------------------------------#      
        
# Tamaño de ventana para el promedio móvil (ancho del kernel).
window_size = 5000 #Es el que más me conviene.

#-----------------------------------------------------------------------#

# KERNEL RECTANGULAR

kernel = np.ones(window_size) / window_size # Promedio Simple.

#-----------------------------------------------------------------------#

# Se plotea con el mismo código de arriba, solo añade dos lineas de código.

fig, axs = plt.subplots(3, 2, figsize=(20,10))  # 3 filas, 2 columnas.

# Loop de cada gas.
for gas_index, gas in enumerate(gases):
    row = gas_index // 2
    col = gas_index % 2
    
    # Loop de cada región.
    for i, region in enumerate(regions):
        # Crea un nombre de columna combinando la región y el gas.
        column_name = f'{region} {gas}'
        
        # Uso la data interpolada.
        data = combined_data[column_name].interpolate()
        
        # Aplico el convolve para smoothear el data. 
        smoothed_data = np.convolve(data, kernel, 'same')   

        # Plot de los datos para dicha región y gas, con etiqueta y color.
        axs[row, col].plot(combined_data['Fecha'], smoothed_data, color = colors[i], label = region)
    
    axs[row, col].set_title(f'Índice de {gas} en las 5 regiones de la Ciudad de México (2019 - 2022)')
    axs[row, col].set_xlabel('Fecha')
    axs[row, col].set_ylabel('Índice')
    axs[row, col].legend(loc='upper right')

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------#


# KERNEL TRIANGULAR

kernel = np.convolve(np.ones(window_size), np.ones(window_size), 'full') # Crea dos matrices y las convoluciona.
kernel = kernel / kernel.sum()  # Normaliza y se asegura que la suma de los pesos sea 1.


#-----------------------------------------------------------------------#

# Se plotea con el mismo código de arriba, solo añade dos lineas de código.

fig, axs = plt.subplots(3, 2, figsize=(20,10))  # 3 filas, 2 columnas.

# Loop de cada gas.
for gas_index, gas in enumerate(gases):
    row = gas_index // 2
    col = gas_index % 2
    
    # Loop de cada región.
    for i, region in enumerate(regions):
        # Crea un nombre de columna combinando la región y el gas.
        column_name = f'{region} {gas}'
        
        # Uso la data interpolada.
        data = combined_data[column_name].interpolate()
        
        # Aplico el convolve para smoothear el data. 
        smoothed_data = np.convolve(data, kernel, 'same')   

        # Plot de los datos para dicha región y gas, con etiqueta y color.
        axs[row, col].plot(combined_data['Fecha'], smoothed_data, color = colors[i], label = region)
    
    axs[row, col].set_title(f'Índice de {gas} en las 5 regiones de la Ciudad de México (2019 - 2022)')
    axs[row, col].set_xlabel('Fecha')
    axs[row, col].set_ylabel('Índice')
    axs[row, col].legend(loc='upper right')

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------#


# KERNEL GAUSSIANO

def gaussian_kernel(size, sigma=1): # Size = tamaño del kernel, Sigma = Desviación Estándar de la Distribución, usualmente es 1 si no lo dan.
    size = int(size) // 2           # Divide el tamaño del kernel entre 2, ya que el KG es simétrico. 
    x = np.linspace(-size, size, 2*size+1) # Crea una matriz de numeros espaciados uniformemente, incluyendo al cero. El total es 2*size**2. 
    g = np.exp(-(x**2) / (2*sigma**2))  # Aplica la función gaussiana.
    return g / g.sum()   # Se normaliza al igual que el kernel triangular, debe dar igual a 1.

kernel = gaussian_kernel(window_size, sigma=window_size/6) # Llama la función y se lo aplica al tamaño ya definido, la sigma es 5000/6.

#-----------------------------------------------------------------------#

# Se plotea con el mismo código de arriba, solo añade dos lineas de código.

fig, axs = plt.subplots(3, 2, figsize=(20,10))  # 3 filas, 2 columnas.

# Loop de cada gas.
for gas_index, gas in enumerate(gases):
    row = gas_index // 2
    col = gas_index % 2
    
    # Loop de cada región.
    for i, region in enumerate(regions):
        # Crea un nombre de columna combinando la región y el gas.
        column_name = f'{region} {gas}'
        
        # Uso la data interpolada.
        data = combined_data[column_name].interpolate()
        
        # Aplico el convolve para smoothear el data. 
        smoothed_data = np.convolve(data, kernel, 'same')   

        # Plot de los datos para dicha región y gas, con etiqueta y color.
        axs[row, col].plot(combined_data['Fecha'], smoothed_data, color = colors[i], label = region)
    
    axs[row, col].set_title(f'Índice de {gas} en las 5 regiones de la Ciudad de México (2019 - 2022)')
    axs[row, col].set_xlabel('Fecha')
    axs[row, col].set_ylabel('Índice')
    axs[row, col].legend(loc='upper right')

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------#

# HISTOGRAMAS


regions = ['Noroeste', 'Noreste', 'Centro', 'Suroeste', 'Sureste']
gases = ['ozono', 'dioxido de azufre', 'dioxido de nitrogeno', 'monoxido de carbono', 'PM10', 'PM25']
colors = ['red', 'magenta', 'lime', 'teal', 'navy'] 

fig, axs = plt.subplots(3, 2, figsize=(20,10))  # 3 filas, 2 columnas.

# Modifica los ejes a una matriz 1D para iterar.
axs = axs.ravel()

for i, gas in enumerate(gases):
    # Loop de cada región.
    for region, color in zip(regions, colors):
        # Crea un nombre de columna combinando la región y el gas.
        column_name = f'{region} {gas}'
        
        # Plotea el histograma.
        axs[i].hist(combined_data[column_name].dropna(), bins=30, histtype='step', label=region, color=color, alpha=1, linewidth=2)  # dropna() excluye valores faltantes.
    
    axs[i].set_title(f'Histograma del índice {gas} en las 5 regiones de la Ciudad de México (2019 - 2022)')
    axs[i].set_xlabel('Índice')
    axs[i].set_ylabel('Conteo Normalizado')
    axs[i].legend(loc='upper right')

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------#