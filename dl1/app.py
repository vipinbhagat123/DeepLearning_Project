import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the trained model

model=tf.keras.models.load_model('model.h5')

# Load the encoder and scaler

with open('ohe.pkl', 'rb') as file:
    ohe=pickle.load(file)

with open('lg.pkl', 'rb') as file:
    lg=pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## Streamlit app
st.title('Customer Churn Prediction')

st.write("Provide customer details to predict the probability of churn:")


st.markdown(
    """
    <style>
    .main {
        background-color: black;
        color: white;
    }
    h1 {
        color: white;
        text-align: center;
    }
    p {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# User input

geography=st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', lg.classes_)
age=st.slider('Age', 18, 92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit_Score')
estimated_salary=st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products=st.slider('Number of Products', 1, 4)
has_cr_card=st.selectbox('Has Credit Card', [0,1])
is_active_member=st.selectbox('Is Active Member', [0,1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [lg.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# One-hot encode 'Geography'
geo_encoded = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')