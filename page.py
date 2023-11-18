import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.title("Seoul Bike Data Prdeiction")

st.write("### We need some information to predict rented bike count")

hour = st.number_input("Hour",min_value=0, max_value=23, value=12)
temperature = st.number_input("Temperature(°C)", value=15)
humidity = st.number_input("Humidity(%)", value=45)
windspeed = st.number_input("Wind speed (m/s)", value=1.2)

ok = st.button("Calculate Rented Bike Count")

df = pd.read_csv('SeoulBikeData.csv', encoding='unicode-escape')

X = df[['Hour', 'Temperature(°C)', 'Humidity(%)','Wind speed (m/s)']]
y = df['Rented Bike Count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()



    
if ok:
    model.fit(X_train, y_train)
    new_data = pd.DataFrame({'Hour': [hour], 'Temperature(°C)': [temperature], 'Humidity(%)': [humidity], 'Wind speed (m/s)': [windspeed]})
    predicted_count = model.predict(new_data)
    st.subheader(f"Estimated number of bicycles rental: {predicted_count[0]:.2f}")