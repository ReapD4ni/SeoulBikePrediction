import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


page = st.sidebar.selectbox("Predict or Explore", ("Predict", "Explore"))





def show_predict_page():
    st.title("Seoul Bike Data Prdeiction")

    st.write("### We need some information to predict rented bike count")

    hour = st.number_input("Hour",min_value=0, max_value=23, value=12)
    temperature = st.number_input("Temperature(°C)", value=15)
    humidity = st.number_input("Humidity(%)", value=45)
    windspeed = st.number_input("Wind speed (m/s)", value=1.2)

    radio=st.radio("Choose from the models below:",("LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor"))

    ok = st.button("Calculate Rented Bike Count")

    df = pd.read_csv('SeoulBikeData.csv', encoding='unicode-escape')

    X = df[['Hour', 'Temperature(°C)', 'Humidity(%)','Wind speed (m/s)']]
    y = df['Rented Bike Count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if(radio == "RandomForestRegressor"):
        model = RandomForestRegressor(max_depth=8, random_state=0)
    elif( radio == "LinearRegression"):
        model= LinearRegression()
    else:
        model=DecisionTreeRegressor(max_depth=6, random_state=0)



        
    if ok:
        model.fit(X_train, y_train)
        new_data = pd.DataFrame({'Hour': [hour], 'Temperature(°C)': [temperature], 'Humidity(%)': [humidity], 'Wind speed (m/s)': [windspeed]})
        predicted_count = model.predict(new_data)
        st.subheader(f"Estimated number of bicycles rental: {predicted_count[0]:.2f}")

def show_exlpore_page():


    df = pd.read_csv('SeoulBikeData.csv', encoding='unicode-escape')

    st.title("Explore Seoul Bike Data")

    hourly_rentals = df.groupby('Hour')['Rented Bike Count'].mean()
    Max_rental_hrs = hourly_rentals.sort_values(ascending=False)
    st.subheader('Hourly Bike Rental Trends')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly_rentals.index, hourly_rentals.values, marker='o', linestyle='-')
    ax.set(xlabel='Hour of the Day', ylabel='Average Rented Bike Count')
    ax.grid(True)
    st.pyplot(fig)


    columns = ['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Rainfall(mm)', 'Snowfall (cm)', 'Seasons']
    st.subheader('Bikes Rented vs. Weather Conditions')
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for i, column in enumerate(columns):
        ax = axes[i // 2, i % 2]
        ax.scatter(df['Rented Bike Count'], df[column])
        ax.set_xlabel('Rented Bike Count')
        ax.set_ylabel(column)
        ax.set_title('Bikes Rented vs. ' + column)
    fig.tight_layout()
    st.pyplot(fig)


if page == "Predict":
    show_predict_page()
else:
    show_exlpore_page()
