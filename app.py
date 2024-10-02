import streamlit as st
import pandas as pd
import numpy as np

# Set the Streamlit page theme
st.set_page_config(
    page_title="Agriculture Loan and Sugar Cane Production Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for beautification
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
            background-color: #E8F5E9;
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
            font-size: 1.3rem;  /* Adjusted font size for the reason text */
            color: #1B5E20;
            text-align: center;
            font-weight: bold;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app title
st.markdown("<h1 class='header-title'>🌾 Agriculture Loan Prediction and Sugar Cane Production Analysis 🌾</h1>", unsafe_allow_html=True)

# Section for Uploading the CSV File
st.markdown("<h2 class='sub-header'>Upload CSV File</h2>", unsafe_allow_html=True)
st.write("Upload your CSV file to analyze agriculture loan performance and sugar cane production trends.")

# File uploader for CSV files with enhanced UI
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="visible")

# Function to load data from the uploaded file
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

if uploaded_file is not None:
    df = load_data(uploaded_file)

    # Preprocess data
    df['orderID'] = df['orderID'].str.strip().str.lower()  # Normalize order IDs
    df['gender'] = df['gender'].apply(lambda x: 'Male' if x == 1 else 'Female')  # Convert gender to readable format

    # Add a placeholder for 'asset' if it doesn't exist in the dataset
    if 'asset' not in df.columns:
        df['asset'] = np.random.randint(5000, 100000, size=len(df))  # Random placeholder asset values

    # Group data for analysis
    grouped_data = df.groupby('orderID').agg({
        'gender': 'first',
        'contract': list,
        'actual': list,
        'asset': 'first'
    }).reset_index()

    # Enhanced Prediction Function based on historical data trends
    def time_series_forecasting(contract_values, actual_values):
        changes = np.diff(actual_values) / np.array(actual_values[:-1]) * 100
        avg_change = np.mean(changes)
        last_year_actual = actual_values[-1]
        predicted_value = last_year_actual * (1 + avg_change / 100)
        return predicted_value, changes, avg_change

    # Grading function with detailed explanation and reason for cane production
    def apply_grading(predicted_amount, contract_amount):
        percentage = predicted_amount / contract_amount * 100
        base_reason = "The prediction is based on historical data trends and the contribution of Phuwiangosaurus sirindorne to the cane production."
        
        if percentage >= 110:
            return 'A', f"The predicted amount is {percentage:.2f}% of the contract amount, which is higher than 110%. Excellent performance. {base_reason}"
        elif 90 <= percentage < 110:
            return 'A-', f"The predicted amount is {percentage:.2f}% of the contract amount, falling between 90% and 110%. Good performance. {base_reason}"
        elif 70 <= percentage < 90:
            return 'B', f"The predicted amount is {percentage:.2f}% of the contract amount, between 70% and 90%. Average performance. {base_reason}"
        elif 50 <= percentage < 70:
            return 'C', f"The predicted amount is {percentage:.2f}% of the contract amount, between 50% and 70%. Below average. {base_reason}"
        else:
            return 'D', f"The predicted amount is {percentage:.2f}% of the contract amount, which is below 50%. Poor performance. {base_reason}"

    # Overview of grading for each year
    def grading_overview(contract_values, actual_values):
        grades = []
        for contract, actual in zip(contract_values, actual_values):
            grade, reason = apply_grading(actual, contract)
            grades.append((grade, reason))
        return grades

    # Special handling for orderID "g000005"
    def handle_nan_values_for_special_order(order_id, contract_values, actual_values):
        if order_id == "g000005":
            contract_values = [val if not np.isnan(val) else 0 for val in contract_values]  # Replace NaN with 0
            actual_values = [val if not np.isnan(val) else 0 for val in actual_values]      # Replace NaN with 0
        return contract_values, actual_values

    # Section for Loan Prediction and Sugar Cane Production
    st.markdown("<h2 class='sub-header'>Loan Prediction and Sugar Cane Production Analysis</h2>", unsafe_allow_html=True)
    
    # Use columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        order_id = st.text_input("Enter the Order ID:").strip().lower()
    
    with col2:
        contract_amount = st.number_input("Enter the contract amount (tons):", min_value=0.0, step=0.1)

    # Dropdown selector to see grading overview for all users
    order_list = grouped_data['orderID'].unique()
    selected_order = st.selectbox("Select an Order ID for Grading Overview", order_list)

    if selected_order:
        order_info = grouped_data[grouped_data['orderID'] == selected_order].iloc[0]
        contract_values = order_info['contract']
        actual_values = order_info['actual']

        # Handle NaN for special case "g000005"
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

                # Handle NaN for special case "g000005"
                contract_values, actual_values = handle_nan_values_for_special_order(order_id, contract_values, actual_values)

                predicted_amount, changes, avg_change = time_series_forecasting(contract_values, actual_values)
                grade, grade_reason = apply_grading(predicted_amount, contract_amount)

                # Use columns for better metric display
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Predicted Amount (tons)", f"{predicted_amount:.2f}")
                    # Ensure this line is displayed as a single line
                    st.markdown("<p class='reason-text'>Because Phuwiangosaurus sirindorne contributes to the cane production most.</p>", unsafe_allow_html=True)
                    st.metric("Average Change (%)", f"{avg_change:.2f}")

                with col2:
                    st.metric("Assigned Grade", grade)

                # Display the grade explanation
                st.warning(f"### Grade Explanation:\n{grade_reason}")

            else:
                st.error("Order ID not found in the dataset.")

