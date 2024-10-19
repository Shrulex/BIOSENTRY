from sklearn.utils import shuffle

# Batch processing to handle large datasets
batch_size = 10000  # Handle data in batches of 10,000 rows

for start in range(0, len(data_scaled), batch_size):
    end = min(start + batch_size, len(data_scaled))
    batch_data = data_scaled[start:end]
    batch_labels = y_train[start:end]

    # Train your model on the batch
    rf_model.partial_fit(batch_data, batch_labels)

print("Batch processing complete.")
