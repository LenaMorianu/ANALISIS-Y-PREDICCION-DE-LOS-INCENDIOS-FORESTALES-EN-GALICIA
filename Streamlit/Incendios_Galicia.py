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
         

  

# Importar el dataset
def importar_datos():
  url = 'https://raw.githubusercontent.com/TFM123456/Big_Data_and_Data_Science_UCM/main/datos_galicia_limpio.csv'
  return pd.read_csv(url, encoding='ISO-8859-1')


# Modelos de predicción
def modelos():
  x = test_size
  df = importar_datos()
  
  #Nos quedamos solo con las variables importantes para predecir
  var_imp = ['superficie', 'time_ctrl', 'PRECIPITACION', 'time_ext', 'SOL',
       'personal', 'RACHA', 'lng', 'lat', 'Año', 'TMAX', 'medios',
       'PRES_RANGE','causa']
  df = df[var_imp]
  
  #Transformamos la variable target CAUSA a numérica
  df.causa.replace(('causa desconocida','fuego reproducido','intencionado','negligencia','rayo'),(0,1,2,3,4), inplace=True)
  
  
  #Imprimit la correlación
  corr = df.corr()['causa']
  
  #Separar la variable target
  X = df.drop(['causa'], axis=1)
  y = df['causa']
  
  #Crear el dataset de TRAIN y TEST
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split (X, y , test_size = x, random_state = 12345, stratify =y)
  
  #Entrenamiento de modelos
  modelos=['Random Forest', 'Logistic Regression']
  column_names=['Modelo','Accuracy','Precision','Recall','F1','Total_Positivos','Total_Negativos','Falsos_Positivos','Falsos_Negativos','Classifier']
  results =pd.DataFrame(columns=column_names)
  
  for i in range(0, len(modelos)):
    
    if i==0:
      from sklearn.ensemble import RandomForestClassifier
      classifier = RandomForestClassifier(random_state=19, creterion='entropy', class_weight='balanced')
    
    else:
      from sklearn.linear_model import LogisticRegression
      classifier = LogisticRegression(solver='sag',class_weight='balanced')
  
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  
  from sklearn import metrics
  acc = metrics.accuracy_score(y_test, y_pred)*100
  prc = metrics.precision_score(y_test, y_pred)*100
  rec = metrics.recall_score(y_test, y_pred)*100
  f1 =  metrics.f1_score(y_test, y_pred)*100
 
  
  from sklearn import confusion_matrix, plot_confusion_matrix
  cm = confusion_matrix(y_test, y_pred)
  tn, fp, fn, tp = cm.ravel()
  
  
  data = [[models[i], acc, prc, rec, f1, tp, tn, fp, fn, classifier]]
  column_names = ['Modelo','Accuracy','Precision','Recall','F1','Total_Positivos','Total_Negativos','Falsos_Positivos','Falsos_Negativos','Classifier']
  model_results = pd.DataFrame(data=data, columns=column_names)
  results = results.append(model_results, ignore_index=True)
  
  return results, corr



datos_galicia = importar_datos()


#Variables de predicción

st.sidebar.subheader('Valores para prediccióin:')

var1 = st.sidebar.number_input('Superficie', min_value=0, max_value=100, step=5)
var2 = st.sidebar.number_input('Time_ctrl', min_value=0, max_value=100, step=5)
var3 = st.sidebar.selectbox('Precipitación:', ['Si','No'])
var4 = st.sidebar.number_input('Time_ext', min_value=0, max_value=100, step=5)
var5 = st.sidebar.number_input('Sol', min_value=0, max_value=100, step=5)
var6 = st.sidebar.number_input('Personal', min_value=0, max_value=100, step=5)
var7 = st.sidebar.number_input('Racha', min_value=0, max_value=100, step=5)
var8 = st.sidebar.number_input('Longitud', min_value=-10.00, max_value=-6.00, step=0.05)
var9 = st.sidebar.number_input('Latitud', min_value=41.00, max_value=44.00, step=0.05)
var10 = st.sidebar.number_input('Año', min_value=2001, max_value=2015, step=1)
var11 = st.sidebar.number_input('TMAX', min_value=-30, max_value=50, step=1)
var12 = st.sidebar.number_input('Medios', min_value=0, max_value=8, step=5)
var13 = st.sidebar.number_input('PRES_RANGE', min_value=0, max_value=15, step=1)



# Preguntar por el tamaño del dataset de TEST
test_size = st.sidebar.slider(label = 'Elige el tamaño del dataset de TEST (%):',
                              min_value=0,
                              max_value=100,
                              value=15,
                              step=1)


results, corr = modelos()

boton_prediccion = st.sider.button('REALIZAR PREDICCIÓN')





st.sidebar.write('')
st.sidebar.write('**AUTORES:**')
st.sidebar.write('**Alejandra García Mosquera**')
st.sidebar.write('**Jorge Gómez Marco**')
st.sidebar.write('**Ana Hernández Villate**')
st.sidebar.write('**Alex Ilundain**')
st.sidebar.write('**Alicia María López Machado**')
st.sidebar.write('**Lenuta Morianu**')
st.sidebar.write('MÁSTER BIG DATA & DATA SCIENCE')
st.sidebar.write('Madrid - Septiembre 2021')


#################################################################


####### IMAGEN

# image = Image.open('./images/IMG1.jpeg')
 
# st.image(image, caption='Mapa Incendios Galicia', use_column_width=True)          
    
  
  
  
