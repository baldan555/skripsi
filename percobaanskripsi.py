import random
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import mysql.connector
from itertools import cycle
from datetime import date
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import mysql.connector
import math
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Conv1D, GRU, Dense
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.subplots as sp
import gspread
from tensorflow.keras.callbacks import EarlyStopping
import _imp




st.set_page_config(page_icon="kalla aspal icon.png", page_title="Kalla Aspal Dash", layout="wide")



st.sidebar.image("logo kalla aspal.png", output_format="PNG", width=300)
page = st.sidebar.selectbox("Select a Page", ["Dashboard", "TrainPredictions", "Other Page"])

st.markdown(
        f"""
        <style>
        .centered-text {{
            text-align: center;
        }}
        </style>
        <div class="centered-text">
            <h1>Kalla Aspal | Price Analysis Dashboard</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
if page == "Dashboard":
    
    

    connection = mysql.connector.connect(
        host="localhost",
        user="kallaaspal",
        password="kalla",
        database="kallaaspal"
    )

    query = "SELECT * FROM data_argus_2"
    cursor = connection.cursor(dictionary=True)
    cursor.execute(query)
    data = cursor.fetchall()
    df = pd.DataFrame(data)
    cursor.close()

    all_months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    with st.expander("Data Preview"):
        st.dataframe(
            df,
            column_config={"Year": st.column_config.NumberColumn(format="%d")},
        )

    df['Start_date'] = pd.to_datetime(df['Start_date'])
    with st.sidebar:
        st.title("Date Filter")
        start_date = st.date_input("Start Date", 
                                min_value=df['Start_date'].min().date(), 
                                max_value=df['Start_date'].max().date(),
                                value=pd.to_datetime('2021-01-01').date())
        end_date = st.date_input("End Date", 
                            min_value=df['Start_date'].min().date(), 
                            max_value=df['Start_date'].max().date(), 
                            value=df['Start_date'].max().date())

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    
    filtered_df = df[(df['Start_date'] >= start_date) & (df['Start_date'] <= end_date)]

    col1, col_spacer, col2 = st.columns([1, 0.1, 1])

    with col1:
        fig = px.line(filtered_df, x='Start_date', y=['Argus_High', 'Argus_Low', 'Argus_Mid'],
                    title="Argus Prices Over Time", labels={'variable': 'Price Type', 'value': 'Price'},
                    hover_data={'variable': True, 'Start_date': True})
        fig.update_layout(title_x=0.4)  
        st.markdown(
            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
            unsafe_allow_html=True
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_spacer:
        st.write("")

    with col2:
        st.dataframe(
            filtered_df.tail(10),  # Menggunakan fungsi tail(15) untuk menampilkan 15 baris terakhir
            column_config={"Year": st.column_config.NumberColumn(format="%d")},
        )

    

    db_connection = mysql.connector.connect(
        host="localhost",
        user="kallaaspal",
        password="kalla",
        database="kallaaspal"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)



    st.markdown("<h1 style='text-align:center'>FOB Bitumen Predicted Price</h1>", unsafe_allow_html=True)

    

    col1, col2 = st.columns(2)


    with col1:

        query_lstm_high = "SELECT * FROM lstm_predict_high"
        query_lstm_low = "SELECT * FROM lstm_predict_low"

        
        cursor_lstm_high = connection.cursor(dictionary=True)
        cursor_lstm_high.execute(query_lstm_high)
        data_lstm_high = cursor_lstm_high.fetchall()
        df_lstm_high = pd.read_sql(query_lstm_high, con=db_connection)

        cursor_lstm_low = connection.cursor(dictionary=True)
        cursor_lstm_low.execute(query_lstm_low)
        data_lstm_low = cursor_lstm_low.fetchall()
        df_lstm_low = pd.read_sql(query_lstm_low, con=db_connection)

        df_combined = pd.merge(df_lstm_low, df_lstm_high, on='DATE', suffixes=('_low', '_high'))
        df_combined['Predicted_price_mid'] = (df_combined['Predicted_price_low'] + df_combined['Predicted_price_high']) / 2

        df_combined.set_index('DATE', inplace=True)

    
        fig_combined = px.line(df_combined, y=['Predicted_price_low', 'Predicted_price_mid', 'Predicted_price_high'],
                            line_shape='linear',  
                            line_dash_sequence=['solid', 'dot', 'dash'],  
                            labels={'value': 'Predicted Price'},
                            color_discrete_sequence=['darkblue', 'lightblue', 'red'],
                            title='LSTM Model Predicted FOB Bitumen Price')

        fig_combined.add_trace(go.Scatter(x=df_combined.index[15:],
                                    y=df_combined['Predicted_price_low'].iloc[15:],
                                    mode='lines+markers',
                                    line=dict(dash='dot'),
                                    marker=dict(color='darkblue'),
                                    name='Predicted_price_low (dots)'))

        fig_combined.add_trace(go.Scatter(x=df_combined.index[15:],
                                    y=df_combined['Predicted_price_mid'].iloc[15:],
                                    mode='lines+markers',
                                    line=dict(dash='dot'),
                                    marker=dict(color='lightblue'),
                                    name='Predicted_price_low (dots)'))

        fig_combined.add_trace(go.Scatter(x=df_combined.index[15:],
                                    y=df_combined['Predicted_price_high'].iloc[15:],
                                    mode='lines+markers',
                                    line=dict(dash='dot'),
                                    marker=dict(color='red'),
                                    name='Predicted_price_low (dots)'))                            


        fig_combined.update_layout(title_x=0.4)  
        st.markdown(
            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
            unsafe_allow_html=True
        )
        st.plotly_chart(fig_combined, use_container_width=True)

        

        # Combine desired columns from df_combined
        combined_predictions = pd.concat([df_combined['Predicted_price_high'], df_combined['Predicted_price_mid'], df_combined['Predicted_price_low']], axis=1)
        combined_predictions.columns = ['High', 'Mid', 'Low']

        # Apply styling to set the width of the table
        st.markdown(
            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
            unsafe_allow_html=True
        )

        # Display the DataFrame with adjusted styling for width
        st.dataframe(combined_predictions.iloc[15:], width=None)


    with col2:
        query_gru_high = "SELECT * FROM gru_predict_high"
        query_gru_low = "SELECT * FROM gru_predict_low"

        
        cursor_gru_high = connection.cursor(dictionary=True)
        cursor_gru_high.execute(query_gru_high)
        data_gru_high = cursor_gru_high.fetchall()
        df_gru_high = pd.read_sql(query_gru_high, con=db_connection)

        cursor_gru_low = connection.cursor(dictionary=True)
        cursor_gru_low.execute(query_gru_low)
        data_gru_low = cursor_gru_low.fetchall()
        df_gru_low = pd.read_sql(query_gru_low, con=db_connection)

        df_combined_gru = pd.merge(df_gru_low, df_gru_high, on='DATE', suffixes=('_low', '_high'))
        df_combined_gru['Predicted_price_mid'] = (df_combined_gru['Predicted_price_low'] + df_combined_gru['Predicted_price_high']) / 2

        df_combined_gru.set_index('DATE', inplace=True)

    
        fig_combined_gru = px.line(df_combined_gru, y=['Predicted_price_low', 'Predicted_price_mid', 'Predicted_price_high'],
                            line_shape='linear',  
                            line_dash_sequence=['solid', 'dot', 'dash'],  
                            labels={'value': 'Predicted Price'},
                            color_discrete_sequence=['darkblue', 'lightblue', 'red'],
                            title='GRU Model Predicted FOB Bitumen Price')

        fig_combined_gru.add_trace(go.Scatter(x=df_combined_gru.index[15:],
                                    y=df_combined_gru['Predicted_price_low'].iloc[15:],
                                    mode='lines+markers',
                                    line=dict(dash='dot'),
                                    marker=dict(color='darkblue'),
                                    name='Predicted_price_low (dots)'))

        fig_combined_gru.add_trace(go.Scatter(x=df_combined_gru.index[15:],
                                    y=df_combined_gru['Predicted_price_mid'].iloc[15:],
                                    mode='lines+markers',
                                    line=dict(dash='dot'),
                                    marker=dict(color='lightblue'),
                                    name='Predicted_price_low (dots)'))

        fig_combined_gru.add_trace(go.Scatter(x=df_combined.index[15:],
                                    y=df_combined_gru['Predicted_price_high'].iloc[15:],
                                    mode='lines+markers',
                                    line=dict(dash='dot'),
                                    marker=dict(color='red'),
                                    name='Predicted_price_low (dots)'))                            


        fig_combined_gru.update_layout(title_x=0.4)  
        st.markdown(
            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
            unsafe_allow_html=True
        )
        st.plotly_chart(fig_combined_gru, use_container_width=True)

        combined_predictions_conv1d = pd.concat([df_combined_gru['Predicted_price_high'], df_combined_gru['Predicted_price_mid'], df_combined_gru['Predicted_price_low']], axis=1)
        combined_predictions_conv1d.columns = ['High', 'Mid', 'Low']

        st.markdown(
            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
            unsafe_allow_html=True
        )
        st.dataframe(combined_predictions_conv1d.iloc[15:], width=None)


   

   



if page == "TrainPredictions":
    connection = mysql.connector.connect(
        host="localhost",
        user="kallaaspal",
        password="kalla",
        database="kallaaspal"
    )

    query = "SELECT * FROM data_argus_2"
    cursor = connection.cursor(dictionary=True)
    cursor.execute(query)
    data = cursor.fetchall()
    df = pd.DataFrame(data)
    cursor.close()

    all_months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    with st.expander("Data Preview"):
        st.dataframe(
            df,
            column_config={"Year": st.column_config.NumberColumn(format="%d")},
        )

    df['Start_date'] = pd.to_datetime(df['Start_date'])
    with st.sidebar:
        st.title("Date Filter")
        start_date = st.date_input("Start Date", min_value=df['Start_date'].min().date(), max_value=df['Start_date'].max().date(), value=df['Start_date'].min().date())
        end_date = st.date_input("End Date", min_value=df['Start_date'].min().date(), max_value=df['Start_date'].max().date(), value=df['Start_date'].max().date())

        
        
        st.title("Insert New Data")
        max_date = date(2070, 12, 31)
        argus_high = st.number_input("Argus High Price", min_value=0.0, value=0.0, key="argus_high")
        argus_low = st.number_input("Argus Low Price", min_value=0.0, value=0.0, key="argus_low")
        argus_mid = st.number_input("Argus Mid Price", min_value=0.0, value=0.0, key="argus_mid")
        start_date_input = st.date_input("Start Date", min_value=df['Start_date'].min().date(),
                              max_value=max_date, value=df['Start_date'].min().date(),
                              key="start_date_input")
        end_date_input = st.date_input("End Date", min_value=df['Start_date'].min().date(),
                              max_value=max_date, value=df['Start_date'].max().date(),
                              key="end_date_input")

        if st.button("Insert Data"):
            connection = mysql.connector.connect(
                host="localhost",
                user="kallaaspal",
                password="kalla",
                database="kallaaspal"
            )

            cursor = connection.cursor()
            insert_query = "INSERT INTO data_argus_2 (Argus_High, Argus_Low, Argus_Mid, Start_date, End_date) " \
                           "VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(insert_query, (argus_high, argus_low, argus_mid, start_date_input, end_date_input))
            connection.commit()
            cursor.close()
            st.success("Data inserted successfully.")

        st.title("Delete Data")
        selected_rows = st.multiselect("Select Rows to Delete", df['Start_date'])

        if st.button("Delete Selected Rows"):
            connection = mysql.connector.connect(
                host="localhost",
                user="kallaaspal",
                password="kalla",
                database="kallaaspal"
            )
            cursor = connection.cursor()

 
            for start_date in selected_rows:
                delete_query = "DELETE FROM data_argus_2 WHERE Start_date = %s"
                cursor.execute(delete_query, (start_date,))
                connection.commit()

            cursor.close()
            connection.close()
            st.success("Selected rows have been deleted.")

        
        df = df[~df['Start_date'].isin(selected_rows)]

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered_df = df[(df['Start_date'] >= start_date) & (df['Start_date'] <= end_date)]

    fig = px.line(filtered_df, x='Start_date', y=['Argus_High', 'Argus_Low', 'Argus_Mid'],
                title="Argus Prices Over Time", labels={'variable': 'Price Type', 'value': 'Price'},
                hover_data={'variable': False, 'Start_date': False})

    st.markdown(
        f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
        unsafe_allow_html=True
    )
    st.plotly_chart(fig, use_container_width=True)


    st.markdown("<h1 style='text-align:center'>Train Predict Model</h1>", unsafe_allow_html=True)
    model_options = ['LSTM', 'GRU']
    selected_data = st.selectbox("Select target column for training", ['Argus_High', 'Argus_Low'])
    selected_model = st.selectbox("Select the model", model_options)
    pred_week = st.slider("Select the number of Weeks to predict", min_value=1, max_value=30, value=10)

    # Dynamic input for parameters
    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.0001, step=0.0001, format="%.4f")
    num_lstm_layers = st.number_input("Number of LSTM Layers", min_value=1, max_value=5, value=1)
    unitss = st.number_input("Units", min_value=10, max_value=100, value=50, step=10)
    batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=8, step=1)
    num_epochs = st.number_input("Number of Epochs", min_value=10, max_value=500, value=100, step=10)
    activation_function = st.selectbox("Activation Function", ['relu', 'sigmoid', 'tanh'])
    if selected_model == 'LSTM':
        if selected_data == 'Argus_High':
            if st.button("Train"):
                if connection.is_connected():
                    cursor = connection.cursor(dictionary=True)
                    cursor.execute("SELECT * FROM data_argus_2")
                    data = cursor.fetchall()
                    df = pd.DataFrame(data)
                    cursor.close()
                    connection.close()

                    with st.spinner("In Progress..."):
                        trans = ['Start_date', 'End_date']

                        for column in trans:
                            df[column] = df[column].astype(str)
                            df[column] = df[column].str.replace(r'\s+', '', regex=True)
                            df[column] = df[column].str[:4] + '-' + df[column].str[4:7] + df[column].str[7:]

                        df['Start_date'] = pd.to_datetime(df['Start_date'])
                        df['End_date'] = pd.to_datetime(df['End_date'])

                        argus = df[['Start_date', 'Argus_High']]
                        copy_price = argus.copy()
                        del argus['Start_date']
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        argus = scaler.fit_transform(np.array(argus).reshape(-1, 1))

                        training_size = int(len(argus) * 0.80)
                        test_size = len(argus) - training_size
                        train_data, test_data = argus[0:training_size, :], argus[training_size:len(argus), :1]

                        def create_dataset(dataset, time_step=0):
                            dataX, dataY = [], []
                            for i in range(len(dataset) - time_step - 1):
                                a = dataset[i:(i + time_step), 0]
                                dataX.append(a)
                                dataY.append(dataset[i + time_step, 0])
                            return np.array(dataX), np.array(dataY)

                        time_step = 15
                        X_train, y_train = create_dataset(train_data, time_step)
                        X_test, y_test = create_dataset(test_data, time_step)

                        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                        model = Sequential()
                        model.add(LSTM(units=unitss, activation=activation_function, input_shape=(time_step, 1)))
                        for _ in range(num_lstm_layers - 1):
                            model.add(LSTM(units=unitss))
                        model.add(Dense(units=1))

                        optimizer = Adam(learning_rate=learning_rate)
                        model.compile(optimizer=optimizer, loss='mean_squared_error')

                        early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

                        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=[early_stopping])

                        loss = model.evaluate(X_test, y_test)

                        train_predict = model.predict(X_train)
                        test_predict = model.predict(X_test)

                        train_predict = scaler.inverse_transform(train_predict)
                        test_predict = scaler.inverse_transform(test_predict)
                        original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
                        original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

                        look_back = time_step

                        trainPredictPlot = np.empty_like(argus)
                        trainPredictPlot[:, :] = np.nan
                        trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

                        testPredictPlot = np.empty_like(argus)
                        testPredictPlot[:, :] = np.nan
                        testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(argus) - 1, :] = test_predict

                        plotdf = pd.DataFrame({
                            'Start_date': copy_price['Start_date'],
                            'original_price': copy_price['Argus_High'],
                            'train_predicted': trainPredictPlot.reshape(1, -1)[0].tolist(),
                            'test_predicted': testPredictPlot.reshape(1, -1)[0].tolist()
                        })

                        fig = px.line(plotdf, x=plotdf['Start_date'],
                                    y=[plotdf['original_price'], plotdf['train_predicted'], plotdf['test_predicted']],
                                    labels={'value': 'price', '': 'Date'})
                        fig.update_layout(title_text='',
                                        plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='')

                        names = cycle(['Harga Aktual', 'Train predicted price', 'Test predicted price'])
                        fig.for_each_trace(lambda t: t.update(name=next(names), line_width=4,))

                        fig.update_xaxes(showgrid=False)
                        fig.update_yaxes(showgrid=False)

                        st.markdown(
                            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
                            unsafe_allow_html=True
                        )
                        st.markdown("<h1 style='text-align:center'>Train Test Plot</h1>", unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("<h1 style='text-align:center'>Model Evaluation Metrics</h1>", unsafe_allow_html=True)

                        train_rmse = math.sqrt(mean_squared_error(original_ytrain, train_predict))
                        train_mse = mean_squared_error(original_ytrain, train_predict)
                        train_mae = mean_absolute_error(original_ytrain, train_predict)

                        test_rmse = math.sqrt(mean_squared_error(original_ytest, test_predict))
                        test_mse = mean_squared_error(original_ytest, test_predict)
                        test_mae = mean_absolute_error(original_ytest, test_predict)
                            
                        
                        
                        coll1 = st.columns(1)
                        with coll1[0]:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training MAE', 'Testing MAE'],
                                y=[train_mae, test_mae],
                                text=[round(train_mae, 2), round(test_mae, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='MAE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5, 
                            
                            )

                            st.plotly_chart(fig)

                        colll1, colll2 = st.columns(2)
                        
                        with colll1:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training MSE', 'Testing MSE'],
                                y=[train_mse, test_mse],
                                text=[round(train_mse, 2), round(test_mse, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='MSE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5,  
                            
                            )

                            st.plotly_chart(fig)

                        with colll2:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training RMSE', 'Testing RMSE'],
                                y=[train_rmse, test_rmse],
                                text=[round(train_rmse, 2), round(test_rmse, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='RMSE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5,  
                                
                            )

                            st.plotly_chart(fig)



                        st.write('Col 2:')
                        st.subheader("Train Metrics:")
                        st.write(f"RMSE: {train_rmse:.2f}")
                        st.write(f"MSE: {train_mse:.2f}")
                        st.write(f"MAE: {train_mae:.2f}")
                       

                        st.subheader("Test Metrics:")
                        st.write(f"RMSE: {test_rmse:.2f}")
                        st.write(f"MSE: {test_mse:.2f}")
                        st.write(f"MAE: {test_mae:.2f}")
                        

            
                        st.subheader("Prediction Results")
                        st.write("Test Loss:", loss)

                        
                    



                        x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
                        temp_input = list(x_input)
                        temp_input = temp_input[0].tolist()

                        lst_output = []
                        n_steps = time_step
                        i = 0
                        pred_week = pred_week

                        
                        while i < pred_week:
                            if len(temp_input) > time_step:
                                x_input = np.array(temp_input[1:])
                                x_input = x_input.reshape(1, -1)
                                x_input = x_input.reshape((1, n_steps, 1))

                                yhat = model.predict(x_input, verbose=0)
                                temp_input.extend(yhat[0].tolist())
                                temp_input = temp_input[1:]
                                lst_output.extend(yhat.tolist())

                                i = i + 1
                            else:
                                x_input = x_input.reshape((1, n_steps, 1))
                                yhat = model.predict(x_input, verbose=0)
                                temp_input.extend(yhat[0].tolist())
                                lst_output.extend(yhat.tolist())

                                i = i + 1

                        
                        start_date = pd.to_datetime(start_date)
                        end_date = pd.to_datetime(end_date)

                        Start_date = df['Start_date'].sort_values(ascending=False).iloc[16].strftime('%Y-%m-%d')
                        last_week=np.arange(1,time_step+1)
                        day_pred=np.arange(time_step+1,time_step+pred_week+1)
                        print(last_week)
                        print(day_pred)

                        temp_mat = np.empty((len(last_week)+pred_week+1,1))
                        temp_mat[:] = np.nan
                        temp_mat = temp_mat.reshape(1,-1).tolist()[0]
                        

                        last_original_week_value = temp_mat
                        next_predicted_week_value = temp_mat

                        last_original_week_value[0:time_step+1] = scaler.inverse_transform(argus[len(argus)-time_step:]).reshape(1,-1).tolist()[0]
                        
                        next_predicted_week_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
                        
                        conv1dgru_results = {
                            'last_original_week_value': last_original_week_value,
                            'next_predicted_week_value': next_predicted_week_value,
                        }

                        new_pred_plot = pd.DataFrame({
                            'last_original_week_value':last_original_week_value,
                            'next_predicted_week_value':next_predicted_week_value,
                            
                        })

                        names = cycle(['Last 15 week close price','Predicted next 10 week price'])
                        new_pred_plot['Timestamp'] = pd.date_range(start=Start_date, periods=len(last_week)+pred_week+1, freq='w')


                        st.markdown("<h1 style='text-align:center'>Plot Prediction</h1>", unsafe_allow_html=True)

                        fig = px.line(new_pred_plot, x='Timestamp', y=['last_original_week_value', 'next_predicted_week_value'],
                                    labels={'value': 'Stock price'},
                                    title='Plot Prediction')

                        fig.update_layout(plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')

                        st.markdown(
                            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
                            unsafe_allow_html=True
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        
                        timestamps = pd.date_range(start=Start_date, periods=len(last_week) + pred_week + 1, freq='w')
                        prediction_results = pd.DataFrame({
                            'Timestamp': timestamps,
                            'Predicted next 10 week price': next_predicted_week_value
                        })

                        if 'last_week' in prediction_results:
                            prediction_results.drop('last_week', axis=1, inplace=True)

                        prediction_results.dropna(subset=['Predicted next 10 week price'], inplace=True)
                        st.write(prediction_results)

            
                        db_connection = mysql.connector.connect(
                            host="localhost",
                            user="kallaaspal",
                            password="kalla",
                            database="kallaaspal"
                        )

                        cursor = db_connection.cursor()
                        cursor.execute("TRUNCATE TABLE lstm_predict_high")
                        for index, row in prediction_results.iterrows():
                            cursor.execute("INSERT INTO lstm_predict_high (Date, `Predicted_price`) VALUES (%s, %s)",
                                        (row['Timestamp'], row['Predicted next 10 week price']))

                        db_connection.commit()

                        cursor.close()
                        db_connection.close()


                else:
                    st.error("Gagal terhubung ke database, periksa koneksi database")


        elif selected_data == 'Argus_Low':
            if st.button("Train"):
                if connection.is_connected():
                    cursor = connection.cursor(dictionary=True)
                    cursor.execute("SELECT * FROM data_argus_2")
                    data = cursor.fetchall()
                    df = pd.DataFrame(data)
                    cursor.close()
                    connection.close()

                    with st.spinner("In Progress..."):
                        trans = ['Start_date', 'End_date']

                        for column in trans:
                            df[column] = df[column].astype(str)
                            df[column] = df[column].str.replace(r'\s+', '', regex=True)
                            df[column] = df[column].str[:4] + '-' + df[column].str[4:7] + df[column].str[7:]

                        df['Start_date'] = pd.to_datetime(df['Start_date'])
                        df['End_date'] = pd.to_datetime(df['End_date'])

                        argus = df[['Start_date', 'Argus_Low']]
                        copy_price = argus.copy()
                        del argus['Start_date']
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        argus = scaler.fit_transform(np.array(argus).reshape(-1, 1))

                        training_size = int(len(argus) * 0.80)
                        test_size = len(argus) - training_size
                        train_data, test_data = argus[0:training_size, :], argus[training_size:len(argus), :1]

                        def create_dataset(dataset, time_step=0):
                            dataX, dataY = [], []
                            for i in range(len(dataset) - time_step - 1):
                                a = dataset[i:(i + time_step), 0]
                                dataX.append(a)
                                dataY.append(dataset[i + time_step, 0])
                            return np.array(dataX), np.array(dataY)

                        time_step = 15
                        X_train, y_train = create_dataset(train_data, time_step)
                        X_test, y_test = create_dataset(test_data, time_step)

                        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                        model = Sequential()
                        model.add(LSTM(units=unitss, activation=activation_function, input_shape=(time_step, 1)))
                        for _ in range(num_lstm_layers - 1):
                            model.add(LSTM(units=unitss))
                        model.add(Dense(units=1))

                        optimizer = Adam(learning_rate=learning_rate)
                        model.compile(optimizer=optimizer, loss='mean_squared_error')

                        early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

                        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=[early_stopping])

                        loss = model.evaluate(X_test, y_test)

                        train_predict = model.predict(X_train)
                        test_predict = model.predict(X_test)

                        train_predict = scaler.inverse_transform(train_predict)
                        test_predict = scaler.inverse_transform(test_predict)
                        original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
                        original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

                        look_back = time_step

                        trainPredictPlot = np.empty_like(argus)
                        trainPredictPlot[:, :] = np.nan
                        trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

                        testPredictPlot = np.empty_like(argus)
                        testPredictPlot[:, :] = np.nan
                        testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(argus) - 1, :] = test_predict

                        plotdf = pd.DataFrame({
                            'Start_date': copy_price['Start_date'],
                            'original_price': copy_price['Argus_Low'],
                            'train_predicted': trainPredictPlot.reshape(1, -1)[0].tolist(),
                            'test_predicted': testPredictPlot.reshape(1, -1)[0].tolist()
                        })

                        fig = px.line(plotdf, x=plotdf['Start_date'],
                                    y=[plotdf['original_price'], plotdf['train_predicted'], plotdf['test_predicted']],
                                    labels={'value': 'price', '': 'Date'})
                        fig.update_layout(title_text='',
                                        plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='')

                        names = cycle(['Harga Aktual', 'Train predicted price', 'Test predicted price'])
                        fig.for_each_trace(lambda t: t.update(name=next(names), line_width=4,))

                        fig.update_xaxes(showgrid=False)
                        fig.update_yaxes(showgrid=False)

                        st.markdown(
                            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
                            unsafe_allow_html=True
                        )
                        st.markdown("<h1 style='text-align:center'>Train Test Plot</h1>", unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("<h1 style='text-align:center'>Model Evaluation Metrics</h1>", unsafe_allow_html=True)

                        train_rmse = math.sqrt(mean_squared_error(original_ytrain, train_predict))
                        train_mse = mean_squared_error(original_ytrain, train_predict)
                        train_mae = mean_absolute_error(original_ytrain, train_predict)

                        test_rmse = math.sqrt(mean_squared_error(original_ytest, test_predict))
                        test_mse = mean_squared_error(original_ytest, test_predict)
                        test_mae = mean_absolute_error(original_ytest, test_predict)
                        

                    

                        
                        coll1 = st.columns(1)

                        with coll1[0]:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training MAE', 'Testing MAE'],
                                y=[train_mae, test_mae],
                                text=[round(train_mae, 2), round(test_mae, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='MAE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5, 
                            
                            )

                            st.plotly_chart(fig)

                        colll1, colll2 = st.columns(2)
                        
                        with colll1:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training MSE', 'Testing MSE'],
                                y=[train_mse, test_mse],
                                text=[round(train_mse, 2), round(test_mse, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='MSE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5,
                            
                            )

                            st.plotly_chart(fig)

                        with colll2:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training RMSE', 'Testing RMSE'],
                                y=[train_rmse, test_rmse],
                                text=[round(train_rmse, 2), round(test_rmse, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='RMSE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5,  
                                
                            )

                            st.plotly_chart(fig)



                        st.write('Col 2:')
                        st.subheader("Train Metrics:")
                        st.write(f"RMSE: {train_rmse:.2f}")
                        st.write(f"MSE: {train_mse:.2f}")
                        st.write(f"MAE: {train_mae:.2f}")
                       

                        st.subheader("Test Metrics:")
                        st.write(f"RMSE: {test_rmse:.2f}")
                        st.write(f"MSE: {test_mse:.2f}")
                        st.write(f"MAE: {test_mae:.2f}")
                        

            
                        st.subheader("Prediction Results")
                        st.write("Test Loss:", loss)



                        x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
                        temp_input = list(x_input)
                        temp_input = temp_input[0].tolist()

                        lst_output = []
                        n_steps = time_step
                        i = 0
                        pred_week = pred_week

                        
                        while i < pred_week:
                            if len(temp_input) > time_step:
                                x_input = np.array(temp_input[1:])
                                x_input = x_input.reshape(1, -1)
                                x_input = x_input.reshape((1, n_steps, 1))

                                yhat = model.predict(x_input, verbose=0)
                                temp_input.extend(yhat[0].tolist())
                                temp_input = temp_input[1:]
                                lst_output.extend(yhat.tolist())

                                i = i + 1
                            else:
                                x_input = x_input.reshape((1, n_steps, 1))
                                yhat = model.predict(x_input, verbose=0)
                                temp_input.extend(yhat[0].tolist())
                                lst_output.extend(yhat.tolist())

                                i = i + 1


                        start_date = pd.to_datetime(start_date)
                        end_date = pd.to_datetime(end_date)

                        Start_date = df['Start_date'].sort_values(ascending=False).iloc[16].strftime('%Y-%m-%d')

                        last_week=np.arange(1,time_step+1)
                        day_pred=np.arange(time_step+1,time_step+pred_week+1)
                        print(last_week)
                        print(day_pred)

                        temp_mat = np.empty((len(last_week)+pred_week+1,1))
                        temp_mat[:] = np.nan
                        temp_mat = temp_mat.reshape(1,-1).tolist()[0]
                        

                        last_original_week_value = temp_mat
                        next_predicted_week_value = temp_mat

                        last_original_week_value[0:time_step+1] = scaler.inverse_transform(argus[len(argus)-time_step:]).reshape(1,-1).tolist()[0]
                        
                        next_predicted_week_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
                        
                        conv1dgru_results = {
                            'last_original_week_value': last_original_week_value,
                            'next_predicted_week_value': next_predicted_week_value,
                        }

                        new_pred_plot = pd.DataFrame({
                            'last_original_week_value':last_original_week_value,
                            'next_predicted_week_value':next_predicted_week_value,
                            
                        })

                        names = cycle(['Last 15 week close price','Predicted next 10 week price'])
                        new_pred_plot['Timestamp'] = pd.date_range(start=Start_date, periods=len(last_week)+pred_week+1, freq='w')


                        st.markdown("<h1 style='text-align:center'>Plot Prediction</h1>", unsafe_allow_html=True)

                        fig = px.line(new_pred_plot, x='Timestamp', y=['last_original_week_value', 'next_predicted_week_value'],
                                    labels={'value': 'Stock price'},
                                    title='Plot Prediction')

                        fig.update_layout(plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')

                        st.markdown(
                            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
                            unsafe_allow_html=True
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        
                        timestamps = pd.date_range(start=Start_date, periods=len(last_week) + pred_week + 1, freq='w')
                        prediction_results = pd.DataFrame({
                            'Timestamp': timestamps,
                            'Predicted next 10 week price': next_predicted_week_value
                        })

                        if 'last_week' in prediction_results:
                            prediction_results.drop('last_week', axis=1, inplace=True)

                        prediction_results.dropna(subset=['Predicted next 10 week price'], inplace=True)
                        st.write(prediction_results)

                        
                        db_connection = mysql.connector.connect(
                            host="localhost",
                            user="kallaaspal",
                            password="kalla",
                            database="kallaaspal"
                        )

                        cursor = db_connection.cursor()
                        cursor.execute("TRUNCATE TABLE lstm_predict_low")
                        for index, row in prediction_results.iterrows():
                            cursor.execute("INSERT INTO lstm_predict_low (DATE, `Predicted_price`) VALUES (%s, %s)",
                                        (row['Timestamp'], row['Predicted next 10 week price']))

            
                        db_connection.commit()

                        cursor.close()
                        db_connection.close()


                else:
                    st.error("Gagal terhubung ke database, periksa koneksi database")






#LSTM MODEL


    elif selected_model == 'GRU':
        if selected_data == 'Argus_High':
            if st.button("Train"):
                if connection.is_connected():
                    cursor = connection.cursor(dictionary=True)
                    cursor.execute("SELECT * FROM data_argus_2")
                    data = cursor.fetchall()
                    df = pd.DataFrame(data)
                    cursor.close()
                    connection.close()

                    with st.spinner("In Progress..."):
                        trans = ['Start_date', 'End_date']

                        for column in trans:
                            df[column] = df[column].astype(str)
                            df[column] = df[column].str.replace(r'\s+', '', regex=True)
                            df[column] = df[column].str[:4] + '-' + df[column].str[4:7] + df[column].str[7:]

                        df['Start_date'] = pd.to_datetime(df['Start_date'])
                        df['End_date'] = pd.to_datetime(df['End_date'])

                        argus = df[['Start_date', 'Argus_High']]
                        copy_price = argus.copy()
                        del argus['Start_date']
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        argus = scaler.fit_transform(np.array(argus).reshape(-1, 1))

                        training_size = int(len(argus) * 0.80)
                        test_size = len(argus) - training_size
                        train_data, test_data = argus[0:training_size, :], argus[training_size:len(argus), :1]

                        def create_dataset(dataset, time_step=0):
                            dataX, dataY = [], []
                            for i in range(len(dataset) - time_step - 1):
                                a = dataset[i:(i + time_step), 0]
                                dataX.append(a)
                                dataY.append(dataset[i + time_step, 0])
                            return np.array(dataX), np.array(dataY)

                        time_step = 15
                        X_train, y_train = create_dataset(train_data, time_step)
                        X_test, y_test = create_dataset(test_data, time_step)

                        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                        model = Sequential()
                        model.add(GRU(units=unitss, activation=activation_function, input_shape=(time_step, 1)))
                        for _ in range(num_lstm_layers - 1):
                            model.add(LSTM(units=unitss))
                        model.add(Dense(units=1))

                        optimizer = Adam(learning_rate=learning_rate)
                        model.compile(optimizer=optimizer, loss='mean_squared_error')

                        early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

                        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=[early_stopping])

                        loss = model.evaluate(X_test, y_test)

                        train_predict = model.predict(X_train)
                        test_predict = model.predict(X_test)

                        train_predict = scaler.inverse_transform(train_predict)
                        test_predict = scaler.inverse_transform(test_predict)
                        original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
                        original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

                        look_back = time_step

                        trainPredictPlot = np.empty_like(argus)
                        trainPredictPlot[:, :] = np.nan
                        trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

                        testPredictPlot = np.empty_like(argus)
                        testPredictPlot[:, :] = np.nan
                        testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(argus) - 1, :] = test_predict

                        plotdf = pd.DataFrame({
                            'Start_date': copy_price['Start_date'],
                            'original_price': copy_price['Argus_High'],
                            'train_predicted': trainPredictPlot.reshape(1, -1)[0].tolist(),
                            'test_predicted': testPredictPlot.reshape(1, -1)[0].tolist()
                        })

                        fig = px.line(plotdf, x=plotdf['Start_date'],
                                    y=[plotdf['original_price'], plotdf['train_predicted'], plotdf['test_predicted']],
                                    labels={'value': 'price', '': 'Date'})
                        fig.update_layout(title_text='',
                                        plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='')

                        names = cycle(['Harga Aktual', 'Train predicted price', 'Test predicted price'])
                        fig.for_each_trace(lambda t: t.update(name=next(names), line_width=4,))

                        fig.update_xaxes(showgrid=False)
                        fig.update_yaxes(showgrid=False)

                        st.markdown(
                            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
                            unsafe_allow_html=True
                        )
                        st.markdown("<h1 style='text-align:center'>Train Test Plot</h1>", unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("<h1 style='text-align:center'>Model Evaluation Metrics</h1>", unsafe_allow_html=True)

                        train_rmse = math.sqrt(mean_squared_error(original_ytrain, train_predict))
                        train_mse = mean_squared_error(original_ytrain, train_predict)
                        train_mae = mean_absolute_error(original_ytrain, train_predict)

                        test_rmse = math.sqrt(mean_squared_error(original_ytest, test_predict))
                        test_mse = mean_squared_error(original_ytest, test_predict)
                        test_mae = mean_absolute_error(original_ytest, test_predict)
                        

                    

                        
                        coll1 = st.columns(1)

                        with coll1[0]:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training MAE', 'Testing MAE'],
                                y=[train_mae, test_mae],
                                text=[round(train_mae, 2), round(test_mae, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='MAE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5, 
                            
                            )

                            st.plotly_chart(fig)

                        colll1, colll2 = st.columns(2)
                        
                        with colll1:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training MSE', 'Testing MSE'],
                                y=[train_mse, test_mse],
                                text=[round(train_mse, 2), round(test_mse, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='MSE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5,
                            
                            )

                            st.plotly_chart(fig)

                        with colll2:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training RMSE', 'Testing RMSE'],
                                y=[train_rmse, test_rmse],
                                text=[round(train_rmse, 2), round(test_rmse, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='RMSE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5,  
                                
                            )

                            st.plotly_chart(fig)



                        st.write('Col 2:')
                        st.subheader("Train Metrics:")
                        st.write(f"RMSE: {train_rmse:.2f}")
                        st.write(f"MSE: {train_mse:.2f}")
                        st.write(f"MAE: {train_mae:.2f}")
                       

                        st.subheader("Test Metrics:")
                        st.write(f"RMSE: {test_rmse:.2f}")
                        st.write(f"MSE: {test_mse:.2f}")
                        st.write(f"MAE: {test_mae:.2f}")
                        

            
                        st.subheader("Prediction Results")
                        st.write("Test Loss:", loss)



                        x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
                        temp_input = list(x_input)
                        temp_input = temp_input[0].tolist()

                        lst_output = []
                        n_steps = time_step
                        i = 0
                        pred_week = pred_week

                        
                        while i < pred_week:
                            if len(temp_input) > time_step:
                                x_input = np.array(temp_input[1:])
                                x_input = x_input.reshape(1, -1)
                                x_input = x_input.reshape((1, n_steps, 1))

                                yhat = model.predict(x_input, verbose=0)
                                temp_input.extend(yhat[0].tolist())
                                temp_input = temp_input[1:]
                                lst_output.extend(yhat.tolist())

                                i = i + 1
                            else:
                                x_input = x_input.reshape((1, n_steps, 1))
                                yhat = model.predict(x_input, verbose=0)
                                temp_input.extend(yhat[0].tolist())
                                lst_output.extend(yhat.tolist())

                                i = i + 1


                        start_date = pd.to_datetime(start_date)
                        end_date = pd.to_datetime(end_date)

                        Start_date = df['Start_date'].sort_values(ascending=False).iloc[16].strftime('%Y-%m-%d')

                        last_week=np.arange(1,time_step+1)
                        day_pred=np.arange(time_step+1,time_step+pred_week+1)
                        print(last_week)
                        print(day_pred)

                        temp_mat = np.empty((len(last_week)+pred_week+1,1))
                        temp_mat[:] = np.nan
                        temp_mat = temp_mat.reshape(1,-1).tolist()[0]
                        

                        last_original_week_value = temp_mat
                        next_predicted_week_value = temp_mat

                        last_original_week_value[0:time_step+1] = scaler.inverse_transform(argus[len(argus)-time_step:]).reshape(1,-1).tolist()[0]
                        
                        next_predicted_week_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
                        
                        conv1dgru_results = {
                            'last_original_week_value': last_original_week_value,
                            'next_predicted_week_value': next_predicted_week_value,
                        }

                        new_pred_plot = pd.DataFrame({
                            'last_original_week_value':last_original_week_value,
                            'next_predicted_week_value':next_predicted_week_value,
                            
                        })

                        names = cycle(['Last 15 week close price','Predicted next 10 week price'])
                        new_pred_plot['Timestamp'] = pd.date_range(start=Start_date, periods=len(last_week)+pred_week+1, freq='w')


                        st.markdown("<h1 style='text-align:center'>Plot Prediction</h1>", unsafe_allow_html=True)

                        fig = px.line(new_pred_plot, x='Timestamp', y=['last_original_week_value', 'next_predicted_week_value'],
                                    labels={'value': 'Stock price'},
                                    title='Plot Prediction')

                        fig.update_layout(plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')

                        st.markdown(
                            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
                            unsafe_allow_html=True
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        
                        timestamps = pd.date_range(start=Start_date, periods=len(last_week) + pred_week + 1, freq='w')
                        prediction_results = pd.DataFrame({
                            'Timestamp': timestamps,
                            'Predicted next 10 week price': next_predicted_week_value
                        })

                        if 'last_week' in prediction_results:
                            prediction_results.drop('last_week', axis=1, inplace=True)

                        prediction_results.dropna(subset=['Predicted next 10 week price'], inplace=True)
                        st.write(prediction_results)

                        
                        db_connection = mysql.connector.connect(
                            host="localhost",
                            user="kallaaspal",
                            password="kalla",
                            database="kallaaspal"
                        )

                        cursor = db_connection.cursor()
                        cursor.execute("TRUNCATE TABLE gru_predict_high")
                        for index, row in prediction_results.iterrows():
                            cursor.execute("INSERT INTO gru_predict_high (DATE, `Predicted_price`) VALUES (%s, %s)",
                                        (row['Timestamp'], row['Predicted next 10 week price']))

            
                        db_connection.commit()

                        cursor.close()
                        db_connection.close()


                else:
                    st.error("Gagal terhubung ke database, periksa koneksi database")

       
        
        



        if selected_data == 'Argus_Low':
            if st.button("Train"):
                if connection.is_connected():
                    cursor = connection.cursor(dictionary=True)
                    cursor.execute("SELECT * FROM data_argus_2")
                    data = cursor.fetchall()
                    df = pd.DataFrame(data)
                    cursor.close()
                    connection.close()

                    with st.spinner("In Progress..."):
                        trans = ['Start_date', 'End_date']

                        for column in trans:
                            df[column] = df[column].astype(str)
                            df[column] = df[column].str.replace(r'\s+', '', regex=True)
                            df[column] = df[column].str[:4] + '-' + df[column].str[4:7] + df[column].str[7:]

                        df['Start_date'] = pd.to_datetime(df['Start_date'])
                        df['End_date'] = pd.to_datetime(df['End_date'])

                        argus = df[['Start_date', 'Argus_Low']]
                        copy_price = argus.copy()
                        del argus['Start_date']
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        argus = scaler.fit_transform(np.array(argus).reshape(-1, 1))

                        training_size = int(len(argus) * 0.80)
                        test_size = len(argus) - training_size
                        train_data, test_data = argus[0:training_size, :], argus[training_size:len(argus), :1]

                        def create_dataset(dataset, time_step=0):
                            dataX, dataY = [], []
                            for i in range(len(dataset) - time_step - 1):
                                a = dataset[i:(i + time_step), 0]
                                dataX.append(a)
                                dataY.append(dataset[i + time_step, 0])
                            return np.array(dataX), np.array(dataY)

                        time_step = 15
                        X_train, y_train = create_dataset(train_data, time_step)
                        X_test, y_test = create_dataset(test_data, time_step)

                        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                        model = Sequential()
                        model.add(GRU(units=unitss, activation=activation_function, input_shape=(time_step, 1)))
                        for _ in range(num_lstm_layers - 1):
                            model.add(LSTM(units=unitss))
                        model.add(Dense(units=1))

                        optimizer = Adam(learning_rate=learning_rate)
                        model.compile(optimizer=optimizer, loss='mean_squared_error')

                        early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

                        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=[early_stopping])

                        loss = model.evaluate(X_test, y_test)

                        train_predict = model.predict(X_train)
                        test_predict = model.predict(X_test)

                        train_predict = scaler.inverse_transform(train_predict)
                        test_predict = scaler.inverse_transform(test_predict)
                        original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
                        original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

                        look_back = time_step

                        trainPredictPlot = np.empty_like(argus)
                        trainPredictPlot[:, :] = np.nan
                        trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

                        testPredictPlot = np.empty_like(argus)
                        testPredictPlot[:, :] = np.nan
                        testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(argus) - 1, :] = test_predict

                        plotdf = pd.DataFrame({
                            'Start_date': copy_price['Start_date'],
                            'original_price': copy_price['Argus_Low'],
                            'train_predicted': trainPredictPlot.reshape(1, -1)[0].tolist(),
                            'test_predicted': testPredictPlot.reshape(1, -1)[0].tolist()
                        })

                        fig = px.line(plotdf, x=plotdf['Start_date'],
                                    y=[plotdf['original_price'], plotdf['train_predicted'], plotdf['test_predicted']],
                                    labels={'value': 'price', '': 'Date'})
                        fig.update_layout(title_text='',
                                        plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='')

                        names = cycle(['Harga Aktual', 'Train predicted price', 'Test predicted price'])
                        fig.for_each_trace(lambda t: t.update(name=next(names), line_width=4,))

                        fig.update_xaxes(showgrid=False)
                        fig.update_yaxes(showgrid=False)

                        st.markdown(
                            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
                            unsafe_allow_html=True
                        )
                        st.markdown("<h1 style='text-align:center'>Train Test Plot</h1>", unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("<h1 style='text-align:center'>Model Evaluation Metrics</h1>", unsafe_allow_html=True)

                        train_rmse = math.sqrt(mean_squared_error(original_ytrain, train_predict))
                        train_mse = mean_squared_error(original_ytrain, train_predict)
                        train_mae = mean_absolute_error(original_ytrain, train_predict)

                        test_rmse = math.sqrt(mean_squared_error(original_ytest, test_predict))
                        test_mse = mean_squared_error(original_ytest, test_predict)
                        test_mae = mean_absolute_error(original_ytest, test_predict)
                        

                    

                        
                        coll1 = st.columns(1)

                        with coll1[0]:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training MAE', 'Testing MAE'],
                                y=[train_mae, test_mae],
                                text=[round(train_mae, 2), round(test_mae, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='MAE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5, 
                            
                            )

                            st.plotly_chart(fig)

                        colll1, colll2 = st.columns(2)
                        
                        with colll1:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training MSE', 'Testing MSE'],
                                y=[train_mse, test_mse],
                                text=[round(train_mse, 2), round(test_mse, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='MSE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5,
                            
                            )

                            st.plotly_chart(fig)

                        with colll2:
                            fig = go.Figure()

                            fig.add_trace(go.Bar(
                                x=['Training RMSE', 'Testing RMSE'],
                                y=[train_rmse, test_rmse],
                                text=[round(train_rmse, 2), round(test_rmse, 2)],
                                textposition='auto',
                                marker=dict(color=['rgb(31, 119, 180)', 'rgb(174, 199, 232)'])
                            ))

                            fig.update_layout(
                                title='RMSE Score',
                                xaxis_title='',
                                yaxis_title='',
                                title_x=0.5,  
                                
                            )

                            st.plotly_chart(fig)



                        st.write('Col 2:')
                        st.subheader("Train Metrics:")
                        st.write(f"RMSE: {train_rmse:.2f}")
                        st.write(f"MSE: {train_mse:.2f}")
                        st.write(f"MAE: {train_mae:.2f}")
                       

                        st.subheader("Test Metrics:")
                        st.write(f"RMSE: {test_rmse:.2f}")
                        st.write(f"MSE: {test_mse:.2f}")
                        st.write(f"MAE: {test_mae:.2f}")
                        

            
                        st.subheader("Prediction Results")
                        st.write("Test Loss:", loss)



                        x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
                        temp_input = list(x_input)
                        temp_input = temp_input[0].tolist()

                        lst_output = []
                        n_steps = time_step
                        i = 0
                        pred_week = pred_week

                        
                        while i < pred_week:
                            if len(temp_input) > time_step:
                                x_input = np.array(temp_input[1:])
                                x_input = x_input.reshape(1, -1)
                                x_input = x_input.reshape((1, n_steps, 1))

                                yhat = model.predict(x_input, verbose=0)
                                temp_input.extend(yhat[0].tolist())
                                temp_input = temp_input[1:]
                                lst_output.extend(yhat.tolist())

                                i = i + 1
                            else:
                                x_input = x_input.reshape((1, n_steps, 1))
                                yhat = model.predict(x_input, verbose=0)
                                temp_input.extend(yhat[0].tolist())
                                lst_output.extend(yhat.tolist())

                                i = i + 1


                        start_date = pd.to_datetime(start_date)
                        end_date = pd.to_datetime(end_date)

                        Start_date = df['Start_date'].sort_values(ascending=False).iloc[16].strftime('%Y-%m-%d')

                        last_week=np.arange(1,time_step+1)
                        day_pred=np.arange(time_step+1,time_step+pred_week+1)
                        print(last_week)
                        print(day_pred)

                        temp_mat = np.empty((len(last_week)+pred_week+1,1))
                        temp_mat[:] = np.nan
                        temp_mat = temp_mat.reshape(1,-1).tolist()[0]
                        

                        last_original_week_value = temp_mat
                        next_predicted_week_value = temp_mat

                        last_original_week_value[0:time_step+1] = scaler.inverse_transform(argus[len(argus)-time_step:]).reshape(1,-1).tolist()[0]
                        
                        next_predicted_week_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
                        
                        conv1dgru_results = {
                            'last_original_week_value': last_original_week_value,
                            'next_predicted_week_value': next_predicted_week_value,
                        }

                        new_pred_plot = pd.DataFrame({
                            'last_original_week_value':last_original_week_value,
                            'next_predicted_week_value':next_predicted_week_value,
                            
                        })

                        names = cycle(['Last 15 week close price','Predicted next 10 week price'])
                        new_pred_plot['Timestamp'] = pd.date_range(start=Start_date, periods=len(last_week)+pred_week+1, freq='w')


                        st.markdown("<h1 style='text-align:center'>Plot Prediction</h1>", unsafe_allow_html=True)

                        fig = px.line(new_pred_plot, x='Timestamp', y=['last_original_week_value', 'next_predicted_week_value'],
                                    labels={'value': 'Stock price'},
                                    title='Plot Prediction')

                        fig.update_layout(plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')

                        st.markdown(
                            f'<style> .css-1ksj3el {{ width: 90%; }} </style>',
                            unsafe_allow_html=True
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        
                        timestamps = pd.date_range(start=Start_date, periods=len(last_week) + pred_week + 1, freq='w')
                        prediction_results = pd.DataFrame({
                            'Timestamp': timestamps,
                            'Predicted next 10 week price': next_predicted_week_value
                        })

                        if 'last_week' in prediction_results:
                            prediction_results.drop('last_week', axis=1, inplace=True)

                        prediction_results.dropna(subset=['Predicted next 10 week price'], inplace=True)
                        st.write(prediction_results)

                        
                        db_connection = mysql.connector.connect(
                            host="localhost",
                            user="kallaaspal",
                            password="kalla",
                            database="kallaaspal"
                        )

                        cursor = db_connection.cursor()
                        cursor.execute("TRUNCATE TABLE gru_predict_low")
                        for index, row in prediction_results.iterrows():
                            cursor.execute("INSERT INTO gru_predict_low (DATE, `Predicted_price`) VALUES (%s, %s)",
                                        (row['Timestamp'], row['Predicted next 10 week price']))

            
                        db_connection.commit()

                        cursor.close()
                        db_connection.close()


                else:
                    st.error("Gagal terhubung ke database, periksa koneksi database")