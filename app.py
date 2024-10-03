import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Agriculture Loan and Sugar Cane Production Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        body {
            background-color: #F7F9F9;
            font-family: 'Arial', sans-serif;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: #F1F8E9;
            color: #1B5E20;
            border: 2px solid #66BB6A;
            font-size: 1.1rem;
            padding: 10px;
            border-radius: 5px;
        }
        .stButton>button {
            background-color: #66BB6A;
            color: white;
            font-size: 1.1rem;
            border-radius: 10px;
            padding: 10px 20px;
            border: 2px solid #4CAF50;
        }
        .stMetric {
            background-color: #207d28;
            padding: 10px;
            border-radius: 10px;
            color: #2E7D32;
            border: 2px solid #66BB6A;
        }
        .stWarning {
            font-size: 1.1rem;
            font-weight: bold;
            background-color: #FFF3E0;
            color: #FFA000;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }
        .header-title {
            color: #1B5E20;
            font-size: 2.5rem;
            text-align: center;
            font-weight: bold;
        }
        .sub-header {
            font-size: 1.8rem;
            font-weight: bold;
            color: #388E3C;
            margin-bottom: 15px;
            text-align: center;
        }
        .table-container {
            margin-top: 20px;
            margin-bottom: 20px;
            background-color: #F1F8E9;
            border: 2px solid #66BB6A;
            border-radius: 10px;
            padding: 20px;
        }
        .table-title {
            font-size: 1.5rem;
            font-weight: bold;
            color: #388E3C;
            margin-bottom: 10px;
        }
        .stTable td, .stTable th {
            font-size: 1.1rem;
            padding: 12px;
        }
        .reason-text {
            font-size: 1.3rem;
            color: #FFFFFF;
            text-align: center;
            font-weight: bold;
            margin-top: 10px;
            background-color: #388E3C;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='header-title'>🌾 Agriculture Loan Prediction and Sugar Cane Production Analysis 🌾</h1>", unsafe_allow_html=True)

st.markdown("<h2 class='sub-header'>Upload CSV File</h2>", unsafe_allow_html=True)
st.write("Upload your CSV file to analyze agriculture loan performance and sugar cane production trends.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="visible")

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

if uploaded_file is not None:
    df = load_data(uploaded_file)

    df['orderID'] = df['orderID'].str.strip().str.lower()
    df['gender'] = df['gender'].apply(lambda x: 'Male' if x == 1 else 'Female')

    if 'asset' not in df.columns:
        df['asset'] = np.random.randint(5000, 100000, size=len(df))

    grouped_data = df.groupby('orderID').agg({
        'gender': 'first',
        'contract': list,
        'actual': list,
        'asset': 'first'
    }).reset_index()

    def train_random_forest_model(df):
        # Prepare data for training
        features = df[['contract', 'asset']]
        target = df['actual']
        
        # Flatten the lists
        features = pd.DataFrame({
            'contract': [item for sublist in features['contract'] for item in sublist],
            'asset': [item for sublist in features['asset'] for item in sublist]
        })
        target = [item for sublist in target for item in sublist]
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        # Train the Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions and evaluate
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        st.write(f"Mean Absolute Error of the model: {mae:.2f}")

        return model

    model = train_random_forest_model(df)

    st.markdown("<h2 class='sub-header'>Loan Prediction and Sugar Cane Production Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        order_id = st.text_input("Enter the Order ID:").strip().lower()
    
    with col2:
        contract_amount = st.number_input("Enter the contract amount (tons):", min_value=0.0, step=0.1)

    order_list = grouped_data['orderID'].unique()
    selected_order = st.selectbox("Select an Order ID for Grading Overview", order_list)

    if selected_order:
        order_info = grouped_data[grouped_data['orderID'] == selected_order].iloc[0]
        contract_values = order_info['contract']
        actual_values = order_info['actual']

        contract_values, actual_values = handle_nan_values_for_special_order(selected_order, contract_values, actual_values)

        grades = grading_overview(contract_values, actual_values)

        year_range = list(range(2015, 2015 + len(contract_values)))
        year_data = pd.DataFrame({
            'Year': year_range,
            'Contract (tons)': contract_values,
            'Actual (tons)': actual_values,
            'Grade': [grade for grade, _ in grades],
            'Reason': [reason for _, reason in grades]
        })

        st.markdown(f"### Grading Overview for Order ID: **{selected_order.upper()}** (2015-2023)")
        st.table(year_data.style.format(na_rep="N/A"))

        if st.button("Predict and Analyze"):
            if order_id in grouped_data['orderID'].values:
                order_info = grouped_data[grouped_data['orderID'] == order_id].iloc[0]
                contract_values = order_info['contract']
                actual_values = order_info['actual']

                contract_values, actual_values = handle_nan_values_for_special_order(order_id, contract_values, actual_values)

                # Prepare features for prediction
                features = pd.DataFrame({
                    'contract': [np.mean(contract_values)],  # Average contract amount
                    'asset': [order_info['asset']]
                })
                predicted_amount = model.predict(features)[0]
                
                grade, grade_reason = apply_grading(predicted_amount, contract_amount)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Predicted Amount (tons)", f"{predicted_amount:.2f}")

                with col2:
                    st.metric("Assigned Grade", grade)
                st.markdown("<p class='reason-text'>Because Phuwiangosaurus sirindorne contributes to the cane production most.</p>", unsafe_allow_html=True)
                st.warning(f"### Grade Explanation:\n{grade_reason}")

            else:
                st.error("Order ID not found in the dataset.")
