#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import numpy as np
import bokeh
from bokeh.plotting import figure

st.set_page_config(
    page_title="Incendios en Galicia",
    #page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded")


st.title('Análisis de los incendios en Galicia - años 2001 - 2015')

DATA_URL = 'https://raw.githubusercontent.com/LenaMorianu/TFM/main/Galicia_definitivo.csv'

DATE_COLUMN = 'fecha'

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


data_load_state = st.text('Loading data...')
data = load_data(100)
data_load_state.text("Done! (using st.cache)")

add_selectbox = st.sidebar.selectbox(
    'Elige la variable que quieres ver:',
    data.columns.values
    )

#st.subheader('Datos de incendios en Galicia')
#with st.container():
 #   if st.checkbox('Muestra los datos'):
  #      st.write(data)
    
#st.dataframe(data.style.highlight_max(axis=0))

left_column, right_column = st.columns(2)


with left_column:
    df2 = data['superficie'].copy()
    st._arrow_line_chart(df2)


with right_column:    
    st.subheader('Datos de incendios en Galicia')
    if st.checkbox('Muestra los datos'):
        st.write(data)
    
    
x = data['superficie']
y = data['fecha']
p = figure(
    title='Evoluciòn superficie quemada',
    x_axis_label='Superficie quemada',
    y_axis_label='Fecha')

p.line(x, y, legend_label='Evolución', line_width=2)
st.bokeh_chart(p, use_container_width=True)

df2 = data.copy()
df2.rename(columns={'lng':'lon'}, inplace=True)

st.map(df2)


st.write(st.session_state)

# st.write(st.session_state['value']) - ERROR