import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

import pickle
import streamlit as st
import shap
import xgboost as xgb

from churn_app_functions import model_performance, roc_score_auc_curve, plot_calibration_curve, plot_learning_curve

# Configure the page to use the full width
st.set_page_config(page_title="Image Display", layout="wide")


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
    
    # Recreate the dataframe
    all_columns = [
        'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 
        'SatisfactionScore', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear', 
        'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount', 'PreferredLoginDevice_Phone', 
        'PreferredPaymentMode_COD', 'PreferredPaymentMode_DC', 'PreferredPaymentMode_UPI', 
        'Gender_Male', 'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop_adons', 'PreferedOrderCat_Mobile',
        'PreferedOrderCat_Others','MaritalStatus_Married', 'MaritalStatus_Single', 'KMeansCluster'
    ]
    
    modelling_data = pd.DataFrame(data_final, columns=all_columns)
    
    return modelling_data

# Load the data
# data_dir = '/home/flame/Desktop/Projects/Churn Prediction/data/'
x_train = pd.read_csv('data/x_train.csv')
x_test = pd.read_csv('data/x_test.csv')
y_train = pd.read_csv('data/y_train.csv')
y_test = pd.read_csv('data/y_test.csv')

# Function to show the home page
def show_home_page():
    st.title("Churn Prediction Home") 
    st.write("")
    st.write("")
    st.header('Input Features')


    col1, col2, col3 = st.columns(3)
    input_data = {}

    with col1:
        input_data['Tenure'] = st.number_input('Tenure of Customer', min_value=0, max_value=70, value=5)
        input_data['CityTier'] = st.radio('Select the City Tier', [1, 2, 3])
        input_data['WarehouseToHome'] = st.number_input('Warehouse to Home Distance', min_value=1, max_value=200, value=30)
        input_data['HourSpendOnApp'] = st.slider('Hours Spent on the App', min_value=0, max_value=10, value=2)
        input_data['NumberOfDeviceRegistered'] = st.slider('Number of Registered Devices', min_value=1, max_value=10)
        input_data['SatisfactionScore'] = st.select_slider('Satisfaction Score', [1, 2, 3, 4, 5], value=4)

    with col2:
        input_data['NumberOfAddress'] = st.number_input('Number of Address', min_value=1, max_value=30, value=2)
        input_data['Complain'] = st.radio('Any Complaints', [0, 1])
        input_data['OrderAmountHikeFromlastYear'] = st.number_input('Order Amount Hike from Last Year', min_value=1, max_value=50, value=15)
        input_data['CouponUsed'] = st.number_input('Number of Coupons Used', min_value=1, max_value=50, value=2)
        input_data['OrderCount'] = st.number_input('Number of Orders', min_value=1, max_value=30, value=2)
        input_data['DaySinceLastOrder'] = st.number_input('Number of Days since last order', min_value=1, max_value=100, value=10)
        input_data['MaritalStatus'] = st.radio('Marital Status', ['Married', 'Single', 'Divorced'])
        

    with col3:
        input_data['CashbackAmount'] = st.slider('Cashback Amount', min_value=0, max_value=500, value=150)
        input_data['PreferredLoginDevice'] = st.radio('Preferred Login Device', ['Phone', 'Computer'])
        input_data['PreferredPaymentMode'] = st.radio('Preferred Payment Mode', ['COD', 'DC', 'UPI', 'CC'])
        input_data['Gender'] = st.radio('Gender', ['Male', 'Female'])
        input_data['PreferedOrderCat'] = st.radio('Preferred Order Category', ['Grocery', 'Laptop_adons', 'Mobile', 'Others', 'Fashion'])


    if st.button('Predict'):
        input_df = pd.DataFrame([input_data])

        # Define categorical columns and their mappings
        categorical_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
        one_hot_mappings = {
            'PreferredLoginDevice': ['PreferredLoginDevice_Phone'],
            'PreferredPaymentMode': ['PreferredPaymentMode_COD', 'PreferredPaymentMode_DC', 'PreferredPaymentMode_UPI', 'PreferredPaymentMode_CC'],
            'Gender': ['Gender_Male'],
            'PreferedOrderCat': ['PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop_adons', 'PreferedOrderCat_Mobile', 'PreferedOrderCat_Others', 'PreferedOrderCat_Fashion'],
            'MaritalStatus': ['MaritalStatus_Married', 'MaritalStatus_Single', 'MaritalStatus_Divorced']
        }

        # Apply one-hot encoding and ensure all required columns are present
        for col in categorical_cols:
            for category in one_hot_mappings[col]:
                input_df[category] = (input_df[col] == category.split('_')[1]).astype(int)
            input_df.drop(columns=[col], inplace=True)
        
        # Ensure all columns are present
        required_columns = [
            'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
            'SatisfactionScore', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear',
            'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount', 'PreferredLoginDevice_Phone',
            'PreferredPaymentMode_COD', 'PreferredPaymentMode_DC', 'PreferredPaymentMode_UPI',
            'Gender_Male', 'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop_adons', 'PreferedOrderCat_Mobile',
            'PreferedOrderCat_Others', 'MaritalStatus_Married', 'MaritalStatus_Single'
        ]
        
        for col in required_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        st.write("")  # Add a line space
        st.write("")
        # st.write("")
        # st.subheader('User Input Values')
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.write(f"**Tenure**: {input_data['Tenure']}")
        #     st.write(f"**CityTier**: {input_data['CityTier']}")
        #     st.write(f"**WarehouseToHome**: {input_data['WarehouseToHome']}")
        #     st.write(f"**HourSpendOnApp**: {input_data['HourSpendOnApp']}")
        #     st.write(f"**NumberOfDeviceRegistered**: {input_data['NumberOfDeviceRegistered']}")
        #     st.write(f"**SatisfactionScore**: {input_data['SatisfactionScore']}")
        #     st.write(f"**NumberOfAddress**: {input_data['NumberOfAddress']}")
        #     st.write(f"**Complain**: {input_data['Complain']}")
        #     st.write(f"**OrderAmountHikeFromlastYear**: {input_data['OrderAmountHikeFromlastYear']}")
        #     st.write(f"**CouponUsed**: {input_data['CouponUsed']}")

        # with col2:
        #     st.write(f"**OrderCount**: {input_data['OrderCount']}")
        #     st.write(f"**DaySinceLastOrder**: {input_data['DaySinceLastOrder']}")
        #     st.write(f"**CashbackAmount**: {input_data['CashbackAmount']}")
        #     st.write(f"**PreferredLoginDevice**: {input_data['PreferredLoginDevice']}")
        #     st.write(f"**PreferredPaymentMode**: {input_data['PreferredPaymentMode']}")
        #     st.write(f"**Gender**: {input_data['Gender']}")
        #     st.write(f"**PreferedOrderCat**: {input_data['PreferedOrderCat']}")
        #     st.write(f"**MaritalStatus**: {input_data['MaritalStatus']}")

        # Preprocess input data
        input_preprocessed = preprocess_input(input_df)

        st.write("")
        st.write("")  
        st.subheader('Prediction')
        # Make prediction
        prediction = final_model.predict(input_preprocessed)
        prediction_proba = final_model.predict_proba(input_preprocessed)

        # Display prediction
        if prediction[0] == 1:
            st.markdown('<p style="color:red;font-weight:bold; font-size:20px;">Churn</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:green;font-weight:bold; font-size:20px;">No Churn</p>', unsafe_allow_html=True)

        st.write("")  # Add a line space
        st.write("")
        st.subheader('Prediction Probability')
        st.write(f'Probability of Churn: {prediction_proba[0][1]:.4f}')
        st.write(f'Probability of No Churn: {prediction_proba[0][0]:.4f}')

# Function to show the model information page
def show_model_info():
    st.title("Model Information")

    
    st.write("### Model Performance")
    model_performance(final_model, x_train, y_train, x_test, y_test)


    st.write("")  # Add a line space
    st.write("### ROC AUC Curve")
    roc_score_auc_curve(final_model, x_train, y_train, x_test, y_test)


    st.write("")  # Add a line space
    st.write("### K-Fold Cross Validation Scores")
    st.markdown('#### The mean recall for the model after 10 folds cross-validation is 0.8607')
    st.markdown('#### The mean accuracy for the model after 10 folds cross-validation is 0.9602')
    st.markdown('#### The mean precision for the model after 10 folds cross-validation is 0.9037')
    st.markdown('#### The mean f1 for the model after 10 folds cross-validation is 0.8796')
    # k_fold_cross_valscore(final_model, x_train, y_train, folds=10)


    st.write("")  # Add a line space
    st.write("### SHAP Summary Plots")
    col1, col2 = st.columns(2)
    with col1:
        st.image("Plots/shap_summary_plot.png", caption="SHAP Summary Bar", use_column_width=True)
    with col2:    
        st.image("Plots/shap_summary_beeswarm.png", caption="SHAP Beeswarm Plot", use_column_width=True)
    
    

    st.write("")  # Add a line space
    st.write("### Partial Dependence Plots")
    # plot_partial_dependence(final_model, x_test)
    st.image("Plots/XGB_dependence_plot.png", caption="XGBoost Partial Dependence Plot", use_column_width=True)
    

    st.write("")  # Add a line space
    st.write("### Calibration Curve")
    plot_calibration_curve(final_model, x_test, y_test)
    
    
    

    st.write("")  # Add a line space
    st.write("### Learning Curve")
    plot_learning_curve(final_model, x_train, y_train)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model Information"])

if page == "Home":
    show_home_page()
else:
    show_model_info()
