#!/usr/bin/env python
# coding: utf-8


import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bokeh
from bokeh.plotting import figure

st.set_page_config( page_title="Incendios en Galicia",
                   #page_icon="🧊",
                   layout="wide",
                   initial_sidebar_state="expanded")




# Importar el dataset
def importar_datos():
  url = 'https://raw.githubusercontent.com/TFM123456/Big_Data_and_Data_Science_UCM/main/datos_galicia_limpio.csv'
  return pd.read_csv(url, encoding='ISO-8859-1')


# Modelos de predicción
def modelos():
  x = test_size
  datos_galicia = importar_datos()
  
  #Nos quedamos solo con las variables importantes para predecir
  var_imp = ['superficie', 'time_ctrl', 'PRECIPITACION', 'time_ext', 'SOL',
       'personal', 'RACHA', 'lng', 'lat', 'Año', 'TMAX', 'medios',
       'PRES_RANGE','causa']
  datos_galicia = datos_galicia[var_imp]
  
  #Transformamos la variable target CAUSA a numérica
  datos_galicia.causa.replace(('causa desconocida','fuego reproducido','intencionado','negligencia','rayo'),(0,1,2,3,4), inplace=True)
  
  
  #Dibujar el mapa de correlación
  corr = datos_galicia.corr()['causa']

  return corr



datos_galicia = importar_datos()

# Preguntar por el tamaño del dataset de TEST
test_size = st.sidebar.slider(label = 'Elige el tamaño del dataset de TEST (%):',
                              min_value=0,
                              max_value=100,
                              value=15,
                              step=1)



corr = modelos()



#################################################################


####### IMAGEN

# image = Image.open('./images/IMG1.jpeg')
 
# st.image(image, caption='Mapa Incendios Galicia', use_column_width=True)          
    

st.title('Análisis y predicción de incendios en Galicia')
         
st.write('El presente proyecto tiene como objetivo el análisis de los incendios producidos en Galicia durante el periodo 2001 - 2015, ' +
         'así como realizar predicciones de la CAUSA de incendios con las características (datos) que el usuarios desea consultar.')
         

  
  
  
