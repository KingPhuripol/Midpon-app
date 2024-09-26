import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set the Streamlit page theme
st.set_page_config(
    page_title="Agriculture Loan and Sugar Cane Production Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the dataset
@st.cache_data
def load_data():
    file_path = r'D:\Iwanttogocnx\1.csv'
    return pd.read_csv(file_path)

df = load_data()

# Preprocess data
df['orderID'] = df['orderID'].str.strip().str.lower()  # Normalize order IDs
df['gender'] = df['gender'].apply(lambda x: 'Male' if x == 1 else 'Female')  # Convert gender to readable format

# Group data for analysis
grouped_data = df.groupby('orderID').agg({
    'gender': 'first',
    'contract': list,
    'actual': list
}).reset_index()

# Simulate time series prediction function
def time_series_forecasting(contract_amount):
    return contract_amount * 1.05  # Apply uniform prediction logic for all order IDs

# Apply grading logic based on the percentage
def apply_grading(predicted_amount, contract_amount):
    percentage = predicted_amount / contract_amount * 100
    if percentage >= 100:
        return 'A'
    elif 80 <= percentage < 100:
        return 'A-'
    elif 60 <= percentage < 80:
        return 'B'
    elif 40 <= percentage < 60:
        return 'C'
    else:
        return 'D'

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #F4F4F9;  /* Light beige background */
            color: #2E7D32;  /* Earthy green font */
        }
        .stTextInput>div>div>input {
            background-color: #E8F5E9;  /* Light green input box */
            color: #2E7D32;
            border: 2px solid #66BB6A;
            font-size: 1.2rem;
            padding: 10px;
            border-radius: 5px;
        }
        .stNumberInput>div>div>input {
            background-color: #E8F5E9;  /* Light green input box */
            color: #2E7D32;
            border: 2px solid #66BB6A;
            font-size: 1.2rem;
            padding: 10px;
            border-radius: 5px;
        }
        .stButton>button {
            background-color: #66BB6A;  /* Green button */
            color: white;
            font-size: 1.2rem;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stError {
            color: #D32F2F;
            font-size: 1.1rem;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app title
st.title("🌾 Agriculture Loan Prediction and Sugar Cane Production Analysis 🌾")

# Input Section for Loan Prediction and Sugar Cane Production
st.header("Loan Prediction and Sugar Cane Production Analysis")
order_id = st.text_input("Enter the Order ID:").strip().lower()
contract_amount = st.number_input("Enter the contract amount (tons):", min_value=0.0, step=0.1)

if st.button("Predict and Analyze"):
    if order_id in grouped_data['orderID'].values:
        # Fetch the details for the given Order ID
        order_info = grouped_data[grouped_data['orderID'] == order_id].iloc[0]
        contract_values = order_info['contract']
        actual_values = order_info['actual']
        predicted_amount = time_series_forecasting(contract_amount)
        grade = apply_grading(predicted_amount, contract_amount)

        # Display Prediction and Grading
        st.markdown(f"### Prediction Result for Order ID: **{order_id.upper()}**")
        st.metric("Predicted Amount (tons)", f"{predicted_amount:.2f}")
        st.metric("Assigned Grade", grade)
        st.write(f"Based on the prediction, the contract for order **{order_id.upper()}** will likely send **{predicted_amount:.2f} tons** this year with a grade of **{grade}**.")
        
        # Create a year-wise breakdown of contract and actual production
        year_data = pd.DataFrame({
            'Year': [f'Year {i+1}' for i in range(len(contract_values))],
            'Contract (tons)': contract_values,
            'Actual (tons)': actual_values
        })

        # Display Data Details in a formatted table
        st.markdown(f"### Details for Order ID: **{order_id.upper()}**")
        st.write(f"**Gender**: {order_info['gender']}")
        st.table(year_data)

        # Plot for contract and actual values
        years = list(range(1, len(contract_values) + 1))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(years, contract_values, label='Contract (tons)', marker='o', color='green')
        ax.plot(years, actual_values, label='Actual (tons)', marker='o', color='blue')
        ax.set_title(f"Contract vs Actual Production for Order ID: {order_id.upper()}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Quantity (tons)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.error("Order ID not found in the dataset.")

# Footer for the app
st.markdown("---")
st.markdown("_This app supports farmers and agricultural workers by predicting loan contract performance and analyzing sugar cane production._")
