import streamlit as st
from PIL import Image

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional

file = 'favicon.png'
favicon = Image.open(file)

st.set_page_config(
     page_title="AInigma LABS",
     page_icon= 'https://ainigma.com.ar/media/favicon.png',
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.instagram.com/Ainigma.labs',
         'Report a bug': "https://www.ainigma.com.ar",
         'About': "# AInigma LABS. Soluciones a tus problemas!"
     }
 )


'''
# Predicción de las univariable a futuro.

Predicción de las ventas o precios futuros utilizando únicamente datos de las ventas pasadas ordenadas por fechas.

'''
# Sidebar section
file = 'logo.png'
image = Image.open(file)

st.sidebar.image(image)
st.sidebar.write('## AInigma')
st.sidebar.write('### Gracias por confiar en nosotros!')


# FIN sidebar section

df = st.file_uploader(
    "-Por favor ingrese su archivo CSV o XSXL con los datos", type=['csv', 'xslx']
)

st.write('''
    #### Debe contar con:
    * Terminación .csv o .xsxl
    * Una columna con los datos de las ventas pasadas o precios pasados llamada 'Prices'
    * Una columna con las fechas llamada 'Date'
''')

data_valid = False

if df is not None: 
    if (df.name.find('.csv') != -1):
        input_df = pd.read_csv(df)
        st.write('''
            ## CSV cargado correctamente
            ''')
        data_valid = True
        days = st.slider('Cuántos días en el futuro desea predecir? Mayor sea el número, menor será la precisión del modelo', 1, 10, 5)
    else:
        if (df.name.find('.xsxl') != -1):
            input_df = pd.read_excel(df)
            st.write('''
                ## Excel cargado correctamente
            ''')
        else: 
            st.error('''
                # El archivo cargado no tiene extensión .CSV
            ''')
else:
    st.write('''
        ## Esperando que se ingrese un archivo
    ''')


def prepare_data(timeseries_data, n_features):
    X, y =[],[]
    for i in range(len(timeseries_data)):
        # find the end of this pattern
        end_ix = i + n_features
        # check if we are beyond the sequence
        if end_ix > len(timeseries_data)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
        X.append(seq_x)
        y.append(seq_y)

    X, y = np.array(X), np.array(y)
    
    return X, y


if (data_valid == True):
    st.dataframe(input_df)
    start_train = st.button('Comenzar entrenamiento')
    timeseries_data = input_df['Price']
    st.line_chart(timeseries_data)
    n_features = 1
    n_steps = days
    if (start_train):
        st.write('Por favor espere mientras su Inteligencia Artificial es entrenada para los datos que usted subió!')
        # Split data into samples
        X, y = prepare_data(timeseries_data, n_steps)   
        X = X.reshape((X.shape[0], n_steps, n_features))
      
        # define model
        model = Sequential()
        model.add(Bidirectional(LSTM(25, activation='relu'), input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=1000, verbose=1)
        # demonstrate prediction
        st.success('Entrenamiento completado!')
        x_input = timeseries_data[-days:].values
        temp_input=list(x_input)
        lst_output=[]
        i=0
        while(i<days):
            if(len(temp_input)>days):
                x_input= np.array(temp_input[1:])
                #print(x_input)
                x_input = x_input.reshape((1, days, n_features))
                #print(x_input)
                yhat = model.predict(x_input, verbose=0)
                temp_input.append(yhat[0][0])
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.append(yhat[0][0])
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps, n_features))
                yhat = model.predict(x_input, verbose=0)
                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])
                i=i+1
    
        st.write('''
            ## Predicciones:
        ''')

        lst_output = pd.Series(lst_output)
        st.write(lst_output)    
        
