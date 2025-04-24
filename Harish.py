# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# 2. Load your dataset
# Replace 'your_dataset.csv' with your dataset file
df = pd.read_csv('your_dataset.csv')

# 3. Explore and preprocess the data
print(df.head())  # Show first 5 rows
print(df.isnull().sum())  # Check for missing values

# Optional: fill or drop missing values
df.fillna(method='ffill', inplace=True)

# 4. Define features (X) and target variable (y)
# Replace 'target_column' with your actual target column name
X = df.drop('target_column', axis=1)
y = df['target_column']

# 5. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 10. Predict on a new sample (example)
# Replace with new patient data in the same format as your features
# Example: new_patient = np.array([[55, 1, 130, 250, 1, 0, 150, 0, 2.3, 1, 0, 3]])
# scaled_sample = scaler.transform(new_patient)
# prediction = model.predict(scaled_sample)
# print("Predicted class:", prediction)
