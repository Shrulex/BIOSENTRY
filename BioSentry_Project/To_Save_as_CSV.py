# Assuming 'data_scaled' is your preprocessed data
import pandas as pd

# Convert scaled data back to a DataFrame (if necessary)
data_scaled_df = pd.DataFrame(data_scaled, columns=data.columns)

# Save the processed data
data_scaled_df.to_csv('preprocessed_biomarker_data.csv', index=False)
print("Preprocessed data saved successfully.")
