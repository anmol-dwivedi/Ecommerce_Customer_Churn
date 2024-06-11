import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import datetime
import warnings
warnings.filterwarnings('ignore')

import pickle
import streamlit as st
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

#####################################################################################################################################################################################################
# Load the Models
def load_model(path):
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {path}")
        return model
    except Exception as e:
        print(f"Failed to load model from {path}: {e}")
        return None

# Specify the directory where the models are saved
load_dir = os.path.join(os.getcwd(), 'model_exports')

# Define the file paths
kmeans_path = os.path.join(load_dir, 'kmeans_model.pkl')
xgboost_path = os.path.join(load_dir, 'xgboost_model.pkl')
preprocessor_path = os.path.join(load_dir, 'preprocessor.pkl')

# Load the models
kmeans = load_model(kmeans_path)
final_model = load_model(xgboost_path)
preprocessor = load_model(preprocessor_path)

# Function to preprocess input data
def preprocess_input(data):
    data_processed = preprocessor.transform(data)
    
    # K-Means Clustering
    kmeans_labels = kmeans.predict(data_processed).reshape(-1, 1)
    
    # Concatenate Features
    data_final = np.hstack((data_processed, kmeans_labels))
    
    
    # recreate the dataframe
    all_columns = [
    'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 
    'SatisfactionScore', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear', 
    'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount', 'PreferredLoginDevice_Phone', 
    'PreferredPaymentMode_COD', 'PreferredPaymentMode_DC', 'PreferredPaymentMode_UPI', 
    'Gender_Male', 'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop_adons', 'PreferedOrderCat_Mobile',
    'PreferedOrderCat_Others','MaritalStatus_Married', 'MaritalStatus_Single', 'KMeansCluster']
    
    modelling_data = pd.DataFrame(data_final, columns=all_columns)
    
    return modelling_data

#######################################################################################################################################################################################################
st.title(""" # E-commerce Churn Prediction """)
st.write('---')

# Sidebar (User Input Pane)
st.sidebar.title('Input Features')
input_data = {
'Tenure' : st.sidebar.number_input('Tenure of Customer', min_value=0, max_value=70, value=3),
'CityTier': st.sidebar.slider('Select the City Tier', min_value=1, max_value=3, value=2),
'WarehouseToHome' : st.sidebar.number_input('Warehouse to Home Distance', min_value=1, max_value=200, value=30),
'HourSpendOnApp' : st.sidebar.slider('Hours Spent on the App', min_value=0, max_value=10, value=2),
'NumberOfDeviceRegistered' : st.sidebar.slider('Number of Registered Devices', min_value=1, max_value=10),
'SatisfactionScore' : st.sidebar.select_slider('Satisfaction Score', [1,2,3,4,5], value=4),
'NumberOfAddress' : st.sidebar.number_input('Number of Address', min_value=1, max_value=30, value=2),
'Complain' : st.sidebar.radio('Any Complaints', [1,0]),
'OrderAmountHikeFromlastYear' : st.sidebar.number_input('Order Amount Hike from Last Year', min_value=1, max_value=50, value=3),
'CouponUsed' : st.sidebar.number_input('Number of Coupons Used', min_value=0, max_value=50, value=2),
'OrderCount' : st.sidebar.number_input('Number of Orders', min_value=1, max_value=30, value=2),
'DaySinceLastOrder' : st.sidebar.number_input('Number of Days since last order', min_value=1, max_value=100, value=10),
'CashbackAmount' : st.sidebar.slider('Cashback Amount', min_value=0, max_value=500, value=100),
'PreferredLoginDevice' : st.sidebar.radio('Preferred Login Device', ['Phone', 'Computer']),   
'PreferredPaymentMode' : st.sidebar.radio('Preferred Payment Model', ['DC', 'CC','UPI', 'CC']),  
'Gender' : st.sidebar.radio('Gender', ['Female',  'Male']), 
'PreferedOrderCat' : st.sidebar.radio('Preferred Order Category', ['Mobile','Grocery','Laptop_adons', 'Others',  'Fashion']), 
'MaritalStatus' : st.sidebar.radio('Marital Status', ['Single', 'Married','Divorced']) 
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Display the input DataFrame
st.subheader('User Inputs')
st.write(input_df)
# st.write(input_df.columns)
# st.write(input_df.dtypes)

#######################################################################################################################################################################################################

# Define categorical columns and their mappings
categorical_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
one_hot_mappings = {
    'PreferredLoginDevice': ['PreferredLoginDevice_Phone'],
    'PreferredPaymentMode': ['PreferredPaymentMode_COD', 
                             'PreferredPaymentMode_DC',
                             'PreferredPaymentMode_UPI', 
                             'PreferredPaymentMode_CC'],
    'Gender': ['Gender_Male'],
    'PreferedOrderCat': ['PreferedOrderCat_Grocery', 
                         'PreferedOrderCat_Laptop_adons',
                         'PreferedOrderCat_Mobile',
                         'PreferedOrderCat_Others',
                         'PreferedOrderCat_Fashion'],
    'MaritalStatus': ['MaritalStatus_Married', 
                      'MaritalStatus_Single', 
                      'MaritalStatus_Divorced']
}

# Apply one-hot encoding and ensure all required columns are present
for col in categorical_cols:
    for category in one_hot_mappings[col]:
        input_df[category] = (input_df[col] == category.split('_')[1]).astype(int)
    input_df.drop(columns=[col], inplace=True)
    
# st.write(input_df)
# st.write(input_df.columns)
# st.write(input_df.dtypes)
    
######################################################################################################################################################################################################

# Ensure all columns are present
required_columns = [
    'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 
    'SatisfactionScore', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear', 
    'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount', 'PreferredLoginDevice_Phone', 
    'PreferredPaymentMode_COD', 'PreferredPaymentMode_DC', 'PreferredPaymentMode_UPI', 
    'Gender_Male', 'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop_adons', 'PreferedOrderCat_Mobile',
    'PreferedOrderCat_Others','MaritalStatus_Married', 'MaritalStatus_Single']

# for col in required_columns:
#     if col not in input_df.columns:
#         input_df[col] = 0

# Keep only the required columns
input_df = input_df[required_columns]


# st.write(input_df)
# st.write(input_df.columns)
# st.write(input_df.dtypes)
#######################################################################################################################################################################################################

# # Display the input DataFrame
# st.subheader('Processed DataFrame')
# processed_df = input_df
# st.write(processed_df)
# st.write(processed_df.columns)

# # Preprocess input data
input_preprocessed = preprocess_input(input_df)

# # Make prediction
prediction = final_model.predict(input_preprocessed)
prediction_proba = final_model.predict_proba(input_preprocessed)

# Display prediction
st.subheader('Prediction')
prediction_text = 'Customer at Risk of Churn' if prediction[0] == 1 else 'Happy Customer'
prediction_color = 'red' if prediction[0] == 1 else 'green'
st.markdown(f"<h2 style='color:{prediction_color};font-weight:bold;'>{prediction_text}</h2>", unsafe_allow_html=True)
# st.write('Churn' if prediction[0] == 1 else 'No Churn')



st.subheader('Prediction Probability')
st.write(f'Probability of Churn: {prediction_proba[0][1]:.4f}')
st.write(f'Probability of No Churn: {prediction_proba[0][0]:.4f}')
