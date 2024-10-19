 # Save the accuracy and classification report to a file
with open('model_report.txt', 'w') as f:
    f.write(f"Random Forest Model Accuracy: {rf_accuracy * 100:.2f}%\n")
    f.write("Classification Report:\n")
    f.write(report)
    
print("Report saved successfully.")
