#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bokeh
from bokeh.plotting import figure
#import pickle
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix, classification_report

st.set_page_config( page_title="Incendios en Galicia",
                   #page_icon="🧊",
                   layout="wide",
                   initial_sidebar_state="expanded")



st.title('Análisis y predicción de incendios forestales en Galicia')
         
st.write('El presente proyecto tiene como objetivo el análisis de los incendios producidos en Galicia durante el periodo 2001 - 2015, ' +
         'así como realizar predicciones de la CAUSA de incendios con las características (datos) que el usuarios desea consultar.')
         

# Preprocesar el dataset (renombrar columnas, etc.)
url = 'https://raw.githubusercontent.com/LenaMorianu/ANALISIS-Y-PREDICCION-DE-LOS-INCENDIOS-FORESTALES-EN-GALICIA/main/Streamlit/dataset_modelo.csv'
df = pd.read_csv(url, encoding='ISO-8859-1')

df.drop(['Unnamed: 0'], axis=1, inplace=True)


df.rename(columns={'superficie':'Superficie_quemada',
                   'lat':'Latitud',
                   'lng':'Longitud',
                   'time_ctrl':'Tiempo_control',
                   'personal':'Personal',
                   'medios':'Medios',
                   'TMEDIA':'Temperatura_media',
                   'RACHA':'Racha',
                   'SOL':'Sol',
                   'AÃ±o':'Ano',
                   'PRES_RANGE':'Presion',
                   'target':'Causa'}, inplace=True)

df.head()


st.write('')
st.write('')
st.write('')
st.write('El dataset de análisis')
st.write('')
st.table(df.head())  



# Crear los dataset de TRAIN y TEST
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Causa'], axis = 1),
                                                    df['Causa'],
                                                    train_size   = 0.8,
                                                    random_state = 1234,
                                                    shuffle      = True,
                                                    stratify = df['Causa'])



modelo = RandomForestClassifier(bootstrap = True, 
                                criterion= 'entropy', 
                                max_depth=None, 
                                n_estimators=150,
                                class_weight='balanced').fit(X_train, y_train)

st.write("El TEST SCORING: {0:.2f} %".format(100 * modelo.score(X_test, y_test)))

#st.table(plot_confusion_matrix(modelo, X_test, y_test, normalize='true'))


st.write('')
st.write('')
st.write('')


#st.table(classification_report(y_test, y_pred))

y_proba = pd.DataFrame(modelo.predict_proba(X_test))
y_proba.columns = y_proba.columns.map({0:'Intencionado',
                                       1:'Causa_desconocida',
                                       2:'Negligencia',
                                       3:'Fuego_reproducido',
                                       4:'Rayo'}).astype(str)

st.write('La probabilidad de cada observación de pertenecer a las clases de la variable target CAUSA':)
st.write('')
st.table(y_proba(head()))

