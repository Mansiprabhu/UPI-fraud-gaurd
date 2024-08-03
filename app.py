import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Define the Streamlit app
def main():
    st.set_page_config(page_title="UPI Fraud Detection", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
        body {
            background-color: #2c3e50;
            color: #000000;
        }
        .title {
            text-align: center;
            color: #000000;
            font-size: 36px;
            font-weight: bold;
        }
        .description {
            text-align: center;
            color: #bdc3c7;
            font-size: 20px;
            margin-bottom: 20px;
        }
        .section-title {
            color: #000000;
            font-size: 28px;
            font-weight: bold;
            border-bottom: 2px solid #e74c3c;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .dataframe {
            color: #000000;
            background-color: #ecf0f1;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 20px;
        }
        .upload-button {
            background-color: #e74c3c;
            color: #ecf0f1;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }
        .upload-button:hover {
            background-color: #c0392b;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="title">UPI Fraud Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="description">Detect anomalous and potentially fraudulent UPI transactions by analyzing patterns in sender-receiver relationships, transaction amounts, and timestamps.</p>', unsafe_allow_html=True)

    # File uploader widget
    uploaded_file = st.file_uploader("Upload a CSV file with UPI transaction data", type="csv")

    if uploaded_file is not None:
        # Read the dataset
        df = pd.read_csv(uploaded_file)

        st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        st.write("First few rows of the dataset:")
        st.write(df.head().style.set_table_styles(
            [{'selector': 'tr:hover', 'props': [('background-color', '#bdc3c7')]}]
        ))

        # Display dataset statistics
        st.markdown('<p class="section-title">Dataset Statistics</p>', unsafe_allow_html=True)
        st.write(df.describe().style.set_table_styles(
            [{'selector': 'tr:hover', 'props': [('background-color', '#bdc3c7')]}]
        ))

        # Plotting histograms for numerical columns
        st.markdown('<p class="section-title">Statistical Diagrams</p>', unsafe_allow_html=True)
        
        num_columns = df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_columns:
            st.markdown(f"**Histogram for {col}**")
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f'Histogram of {col}', color='white')
            ax.set_xlabel(col, color='white')
            ax.set_ylabel('Frequency', color='white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            st.pyplot(fig)

        # Automatically detect relevant columns
        def detect_columns(df):
            columns = {
                'sender': None,
                'receiver': None,
                'amount': None,
                'timestamp': None
            }
            for col in df.columns:
                col_lower = col.lower()
                if 'sender' in col_lower:
                    columns['sender'] = col
                elif 'receiver' in col_lower:
                    columns['receiver'] = col
                elif 'amount' in col_lower:
                    columns['amount'] = col
                elif 'time' in col_lower:
                    columns['timestamp'] = col
            return columns

        columns = detect_columns(df)

        if not all(columns.values()):
            missing_columns = [key for key, value in columns.items() if value is None]
            st.warning(f"Some columns could not be detected: {', '.join(missing_columns)}. The application will try to proceed with the detected columns.")

        # Ensure required columns are detected
        if columns['sender'] and columns['receiver'] and columns['amount'] and columns['timestamp']:
            st.markdown('<p class="section-title">Fraud Detection</p>', unsafe_allow_html=True)

            # Convert timestamp to datetime
            df[columns['timestamp']] = pd.to_datetime(df[columns['timestamp']])

            # Extract features
            df['hour'] = df[columns['timestamp']].dt.hour
            df['day_of_week'] = df[columns['timestamp']].dt.dayofweek

            # Encode sender and receiver
            df['sender_receiver'] = df[columns['sender']] + '_' + df[columns['receiver']]
            df_encoded = pd.get_dummies(df[['sender_receiver', 'hour', 'day_of_week', columns['amount']]])

            # Anomaly detection using Isolation Forest
            model = IsolationForest(contamination=0.01, random_state=42)
            df['anomaly'] = model.fit_predict(df_encoded)

            # Filter anomalies
            anomalies = df[df['anomaly'] == -1]

            st.markdown('<p class="section-title">Anomalies Detected</p>', unsafe_allow_html=True)
            st.write(f"Number of anomalies: {anomalies.shape[0]}")
            st.write(anomalies.style.set_table_styles(
                [{'selector': 'tr:hover', 'props': [('background-color', '#e74c3c')]}]
            ))

            # Visualization
            st.markdown('<p class="section-title">Anomalies Visualization</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(14, 7))
            sns.scatterplot(data=df, x=columns['amount'], y='hour', hue='anomaly', palette={1: 'blue', -1: 'red'}, ax=ax)
            ax.set_title('Anomalous Transactions', fontsize=22, color='white')
            ax.set_xlabel('Transaction Amount', fontsize=16, color='white')
            ax.set_ylabel('Hour of the Day', fontsize=16, color='white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            st.pyplot(fig)

            # Recommendations to avoid anomalies
            st.markdown('<p class="section-title">Recommendations to Avoid Anomalies</p>', unsafe_allow_html=True)
            st.write("""
                1. **Monitor Unusual Patterns**: Regularly analyze transaction patterns to detect any unusual activity.
                2. **Implement Robust Authentication**: Use strong authentication methods such as OTP, biometric verification, and multi-factor authentication.
                3. **Limit Transaction Amounts**: Set limits on transaction amounts and monitor transactions that exceed these limits.
                4. **Time-based Analysis**: Monitor transactions made during unusual hours or on weekends, as they may be indicative of fraudulent activity.
                5. **Geographic Monitoring**: Track transactions from unusual or unexpected locations.
                6. **Regular Audits**: Conduct regular audits and reviews of transaction data to identify and address any vulnerabilities.
                7. **Education and Awareness**: Educate users about potential fraud risks and encourage them to report suspicious activity.
                8. **Advanced Machine Learning Models**: Continuously improve and update fraud detection models to adapt to new fraud techniques.
            """)

        else:
            st.write("Required columns not found. Please ensure your dataset includes 'sender', 'receiver', 'amount', and 'timestamp' columns.")

if __name__ == "__main__":
    main()
