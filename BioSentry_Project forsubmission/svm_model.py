# svm_model.py
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load preprocessed data
data = pd.read_csv('preprocessed_biomarker_data.csv')

# Split data
X = data.drop(columns=['target'])  # Replace 'target' with your actual label column
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train SVM
svm_model = SVC(kernel='linear', verbose=True)
svm_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(svm_model, 'models/svm_model.pkl')
print("SVM model saved successfully as models/svm_model.pkl.")

# Make predictions and evaluate
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save the report
with open('results/svm_classification_report.txt', 'w') as f:
    f.write(f"SVM Accuracy: {accuracy * 100:.2f}%\n")
    f.write("Classification Report:\n")
    f.write(report)
print("SVM classification report saved as results/svm_classification_report.txt.")
