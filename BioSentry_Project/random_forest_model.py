# random_forest_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load preprocessed data
data = pd.read_csv('preprocessed_biomarker_data.csv')

# Split data
X = data.drop(columns=['target'])  # Replace 'target' with your actual label column
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, 'models/random_forest_model.pkl')
print("Random Forest model saved successfully as models/random_forest_model.pkl.")

# Make predictions and evaluate
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
