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



st.title('Análisis y predicción de incendios en Galicia')
         
st.write('El presente proyecto tiene como objetivo el análisis de los incendios producidos en Galicia durante el periodo 2001 - 2015, ' +
         'así como realizar predicciones de la CAUSA de incendios con las características (datos) que el usuarios desea consultar.')
         


url = 'https://raw.githubusercontent.com/LenaMorianu/ANALISIS-Y-PREDICCION-DE-LOS-INCENDIOS-FORESTALES-EN-GALICIA/main/Streamlit/dataset_modelo.csv'
df = pd.read_csv(url, encoding='ISO-8859-1')
    
st.write('')
st.write('')
st.write('')
st.write('El dataset de análisis')
st.write('********************************************************************************')
st.table(df.head())  
 

