# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load dataset (replace 'biomarker_data.csv' with your actual file)
data = pd.read_csv('biomarker_data.csv')

# Handle missing data
imputer = SimpleImputer(strategy='mean')  # Mean imputation for missing values
data_imputed = imputer.fit_transform(data)

# Encode categorical data (e.g., if there are disease categories)
encoder = LabelEncoder()
data['disease_category'] = encoder.fit_transform(data['disease_category'])

# Feature scaling (normalization)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Convert scaled data back to DataFrame
data_scaled_df = pd.DataFrame(data_scaled, columns=data.columns)

# Save the processed data
data_scaled_df.to_csv('preprocessed_biomarker_data.csv', index=False)
print("Preprocessed data saved successfully as preprocessed_biomarker_data.csv.")
