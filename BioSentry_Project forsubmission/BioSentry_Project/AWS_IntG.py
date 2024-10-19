import boto3

# Initialize the S3 client
s3 = boto3.client('s3')

# Upload the model
s3.upload_file('random_forest_model.pkl', 'your-bucket-name', 'models/random_forest_model.pkl')
print("Model uploaded to S3 successfully.")
