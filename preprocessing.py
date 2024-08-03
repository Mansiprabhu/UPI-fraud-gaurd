import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
file_path = 'C:/Users/admin/Desktop/ML internship/transactions.csv'  # Update this path
df = pd.read_csv(file_path)

# 1. Handling Missing Values
# Check for missing values
missing_values = df.isnull().sum()

# Fill missing values if necessary (depending on the dataset, here we assume no missing values to keep it simple)
# df.fillna(method='ffill', inplace=True) # Example of forward fill

# 2. Handling Duplicates
# Remove duplicate rows
df.drop_duplicates(inplace=True)

# 3. Feature Engineering
# Extract date and time components from the Timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Hour'] = df['Timestamp'].dt.hour
df['Minute'] = df['Timestamp'].dt.minute
df['Second'] = df['Timestamp'].dt.second
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

# Create new features based on sender-receiver relationships (example: number of transactions per sender)
df['Sender Transaction Count'] = df.groupby('Sender UPI ID')['Transaction ID'].transform('count')
df['Receiver Transaction Count'] = df.groupby('Receiver UPI ID')['Transaction ID'].transform('count')

# 4. Encoding Categorical Variables
# Encoding categorical variables using LabelEncoder
label_encoders = {}
categorical_columns = ['Sender Name', 'Sender UPI ID', 'Receiver Name', 'Receiver UPI ID', 'Status']

for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# 5. Scaling Numerical Features
scaler = StandardScaler()
df['Amount (INR)'] = scaler.fit_transform(df[['Amount (INR)']])

# Drop the original Timestamp column as we have extracted useful features from it
df.drop(columns=['Timestamp'], inplace=True)

# Save the preprocessed dataset to a new CSV file
preprocessed_file_path = 'C:/Users/admin/Desktop/ML internship/preprocessed_transactions.csv'  # Update this path
df.to_csv(preprocessed_file_path, index=False)

print("Preprocessing complete. The preprocessed dataset is saved to:", preprocessed_file_path)

