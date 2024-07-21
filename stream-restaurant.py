import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the model
resto_model = pickle.load(open('resto_model.sav', 'rb'))

# Load data (required for fitting label encoders)
data = pd.read_csv('restaurant_menu_optimization_data.csv')
df = data[['RestaurantID', 'MenuCategory', 'MenuItem', 'Price', 'Profitability']]

# Ensure 'Price' is of string type and then convert to float
df['Price'] = df['Price'].astype(str).str.replace(',', '.').astype(float)

# Encode 'Profitability' to numeric values
def encode_Profitability(Profitability):
    if Profitability == 'Low':
        return 0
    elif Profitability == 'Medium':
        return 1
    elif Profitability == 'High':
        return 2
    else:
        return None

df['Profitability'] = df['Profitability'].apply(encode_Profitability)

# Encode categorical variables
categorical_cols = ['RestaurantID', 'MenuCategory', 'MenuItem']
label_encoders = {col: LabelEncoder() for col in categorical_cols}

for col in categorical_cols:
    df[col] = label_encoders[col].fit_transform(df[col])

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
    try:
        # Convert inputs to the required format
        RestaurantID_encoded = label_encoders['RestaurantID'].transform([RestaurantID])[0]
        MenuCategory_encoded = label_encoders['MenuCategory'].transform([MenuCategory])[0]
        MenuItem_encoded = label_encoders['MenuItem'].transform([MenuItem])[0]
        Price = float(Price)

        # Debugging print statements
        st.write(f"Encoded RestaurantID: {RestaurantID_encoded}")
        st.write(f"Encoded MenuCategory: {MenuCategory_encoded}")
        st.write(f"Encoded MenuItem: {MenuItem_encoded}")
        st.write(f"Price: {Price}")

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
    except ValueError as e:
        st.error(f"Error: {e}")
