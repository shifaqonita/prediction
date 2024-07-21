import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the model
resto_model = pickle.load(open('resto_model.sav', 'rb'))

# Load label encoders
label_encoders = {col: LabelEncoder() for col in ['RestaurantID', 'MenuCategory', 'MenuItem']}
for col in label_encoders.keys():
    label_encoders[col].fit(df[col])

# Title of the web app
st.title('Prediksi Profitability')

# Split columns for input
col1, col2 = st.columns(2)

with col1:
    RestaurantID = st.text_input('Input RestaurantID')

with col2:
    MenuCategory = st.text_input('Input MenuCategory')

with col1:
    MenuItem = st.text_input('Input MenuItem')

with col2:
    Price = st.text_input('Input Price')

# Prediction result
resto_pred = ''

# Create a button for prediction
if st.button('Test Prediksi Profitability'):
    # Convert inputs to the required format
    RestaurantID_encoded = label_encoders['RestaurantID'].transform([RestaurantID])[0]
    MenuCategory_encoded = label_encoders['MenuCategory'].transform([MenuCategory])[0]
    MenuItem_encoded = label_encoders['MenuItem'].transform([MenuItem])[0]
    Price = float(Price)

    # Predict
    resto_prediction = resto_model.predict([[RestaurantID_encoded, MenuCategory_encoded, MenuItem_encoded, Price]])

    # Map prediction to the corresponding label
    if resto_prediction[0] == 0:
        resto_pred = 'Low'
    elif resto_prediction[0] == 1:
        resto_pred = 'Medium'
    else:
        resto_pred = 'High'
    st.success(resto_pred)
