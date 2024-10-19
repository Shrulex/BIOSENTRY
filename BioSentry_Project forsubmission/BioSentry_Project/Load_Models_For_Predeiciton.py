# Load the model
loaded_rf_model = joblib.load('random_forest_model.pkl')
# Make predictions with the loaded model
y_pred_loaded = loaded_rf_model.predict(X_test)
