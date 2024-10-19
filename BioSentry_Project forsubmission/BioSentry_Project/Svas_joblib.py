import joblib

# Save Random Forest model
joblib.dump(rf_model, 'random_forest_model.pkl')
print("Random Forest model saved.")

# Save SVM model
joblib.dump(svm_model, 'svm_model.pkl')
print("SVM model saved.")
