# cloud_upload.py
from google.cloud import storage

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

# Example usage for uploading Random Forest model to GCS
upload_to_gcs('your-bucket-name', 'models/random_forest_model.pkl', 'models/random_forest_model.pkl')
