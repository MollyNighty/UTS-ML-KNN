import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load datasets
ds_fishKNN = pd.read_csv("fish.csv")
ds_fruitKNN = pd.read_excel("fruit.xlsx")

# Pilihan dataset
st.title("Aplikasi Prediksi KNN")
pilih_dataset = st.selectbox("Pilih dataset:", ["Fish", "Fruit"])

if pilih_dataset == "Fish":
    
    # Encoding categorical target
    le = LabelEncoder()
    ds_fishKNN['species'] = le.fit_transform(ds_fishKNN['species'])
    
    # Splitting data
    X = ds_fishKNN[['length', 'weight', 'w_l_ratio']]
    y = ds_fishKNN['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # User input for new data
    st.subheader("Prediksi Jenis Ikan")
    length = st.number_input("length:", min_value=0.0)
    weight = st.number_input("weight:", min_value=0.0)
    w_l_ratio = st.number_input("w_l_ratio:", min_value=0.0)

    if st.button("Prediksi"):
        new_data = np.array([[length, weight, w_l_ratio]])
        prediction = knn.predict(new_data)
        species_pred = le.inverse_transform(prediction)
        st.write(f"Hasil prediksi: {species_pred[0]}")
        
elif pilih_dataset == "Fruit":
    
    # Encoding categorical target
    le = LabelEncoder()
    ds_fruitKNN['name'] = le.fit_transform(ds_fruitKNN['name'])
    
    # Splitting data
    X = ds_fruitKNN[['diameter', 'weight', 'red', 'green', 'blue']]
    y = ds_fruitKNN['name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # User input for new data
    st.subheader("Prediksi Jenis Buah")
    diameter = st.number_input("diameter:", min_value=0.0)
    weight = st.number_input("weight:", min_value=0.0)
    red = st.number_input("red:", min_value=0.0)
    green = st.number_input("green:", min_value=0.0)
    blue = st.number_input("blue:", min_value=0.0)

    if st.button("Prediksi"):
        new_data = np.array([[diameter, weight, red, green, blue]])
        prediction = knn.predict(new_data)
        fruit_pred = le.inverse_transform(prediction)
        st.write(f"Hasil prediksi: {fruit_pred[0]}")
