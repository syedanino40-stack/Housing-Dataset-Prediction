
import streamlit as st
import pickle
import numpy as np

# Load Model
with open('Housing_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load Scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Page Config
st.set_page_config(
    page_title="Housing Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 Housing Price Prediction App")

st.markdown("### Enter House Details")

# Columns Layout
col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input("Area", min_value=0)

    bedrooms = st.number_input("Bedrooms", min_value=0)

    bathrooms = st.number_input("Bathrooms", min_value=0)

    stories = st.number_input("Stories", min_value=0)

with col2:
    mainroad = st.selectbox("Main Road", ["yes", "no"])

    guestroom = st.selectbox("Guest Room", ["yes", "no"])

    basement = st.selectbox("Basement", ["yes", "no"])

    hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])

with col3:
    airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])

    parking = st.number_input("Parking", min_value=0)

    prefarea = st.selectbox("Preferred Area", ["yes", "no"])

    furnishingstatus = st.selectbox(
        "Furnishing Status",
        ["furnished", "semi-furnished", "unfurnished"]
    )

# Convert categorical values
mainroad = 1 if mainroad == "yes" else 0
guestroom = 1 if guestroom == "yes" else 0
basement = 1 if basement == "yes" else 0
hotwaterheating = 1 if hotwaterheating == "yes" else 0
airconditioning = 1 if airconditioning == "yes" else 0
prefarea = 1 if prefarea == "yes" else 0

# Furnishing Encoding
if furnishingstatus == "furnished":
    furnishingstatus = 2
elif furnishingstatus == "semi-furnished":
    furnishingstatus = 1
else:
    furnishingstatus = 0

# Prediction Button
if st.button("Predict House Price"):

    features = np.array([[
        area,
        bedrooms,
        bathrooms,
        stories,
        mainroad,
        guestroom,
        basement,
        hotwaterheating,
        airconditioning,
        parking,
        prefarea,
        furnishingstatus
    ]])

    # Scale Features
    features = scaler.transform(features)

    # Prediction
    prediction = model.predict(features)

    st.markdown("---")
    st.subheader("Predicted House Price")

    st.success(f"🏠 Estimated Price: {prediction[0]:,.2f}")
