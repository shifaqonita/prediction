import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Muat model
resto_model = pickle.load(open('resto_model.sav', 'rb'))

# Muat data (diperlukan untuk fitting label encoders)
data = pd.read_csv('restaurant_menu_optimization_data.csv')
df = data[['RestaurantID', 'MenuCategory', 'MenuItem', 'Price', 'Profitability']]

# Pastikan 'Price' bertipe string lalu ubah menjadi float
df['Price'] = df['Price'].astype(str).str.replace(',', '.').astype(float)

# Encode 'Profitability' menjadi nilai numerik
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

# Encode variabel kategorikal
categorical_cols = ['RestaurantID', 'MenuCategory', 'MenuItem']
label_encoders = {col: LabelEncoder() for col in categorical_cols}

for col in categorical_cols:
    df[col] = label_encoders[col].fit_transform(df[col])

# Judul aplikasi web
st.title('Prediksi Profitability')

# Bagi kolom untuk input
col1, col2 = st.columns(2)

with col1:
    RestaurantID = st.text_input('Input RestaurantID')

with col2:
    MenuCategory = st.text_input('Input MenuCategory')

with col1:
    MenuItem = st.text_input('Input MenuItem')

with col2:
    Price = st.text_input('Input Price')

# Hasil prediksi
resto_pred = ''

# Buat tombol untuk prediksi
if st.button('Test Prediksi Profitability'):
    # Ubah input ke format yang diperlukan
    RestaurantID_encoded = label_encoders['RestaurantID'].transform([RestaurantID])[0]
    MenuCategory_encoded = label_encoders['MenuCategory'].transform([MenuCategory])[0]
    MenuItem_encoded = label_encoders['MenuItem'].transform([MenuItem])[0]
    Price = float(Price)

    # Prediksi
    resto_prediction = resto_model.predict([[RestaurantID_encoded, MenuCategory_encoded, MenuItem_encoded, Price]])

    # Peta hasil prediksi ke label yang sesuai
    if resto_prediction[0] == 0:
        resto_pred = 'Low'
    elif resto_prediction[0] == 1:
        resto_pred = 'Medium'
    else:
        resto_pred = 'High'
    st.success(resto_pred)
