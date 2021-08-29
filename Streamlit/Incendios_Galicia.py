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



st.title('                             Análisis y predicción de incendios en Galicia')
         
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
  y = data['causa']
  
  #Crear el dataset de TRAIN y TEST
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=x)
  
  #Entrenamiento de modelos
  modelos=['Random Forest', 'Logistic Regression']
  column_names=['Modelo','Accuracy','Precision','Recall','F1','Total_Positivos','Total_Negativos','Falsos_Positivos','Falsos_Negativos','Classifier']
  results =pd.DataFrame(column=column_names)
  
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

st.write(datos_galicia.head())

# Preguntar por el tamaño del dataset de TEST
test_size = st.sidebar.slider(label = 'Elige el tamaño del dataset de TEST (%):',
                              min_value=0,
                              max_value=100,
                              value=15,
                              step=1)



results, corr = modelos()

st.write(corr, results)



#################################################################


####### IMAGEN

# image = Image.open('./images/IMG1.jpeg')
 
# st.image(image, caption='Mapa Incendios Galicia', use_column_width=True)          
    
  
  
  
