import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, plot_importance
from matplotlib import pyplot as plt

# Load dataset
df = pd.read_csv('C:/Users/rahul/OneDrive/Desktop/SRM/projects/AI/nasa.csv')

# Display dataset information
print(df.shape)
print(df.info())

# Drop unnecessary columns
drop_cols = ['Neo Reference ID', 'Name', 'Orbit ID', 'Close Approach Date',
             'Epoch Date Close Approach', 'Orbit Determination Date']
df = df.drop(drop_cols, axis=1)

# Encode categorical variable "Hazardous" using one-hot encoding
df['Hazardous'] = df['Hazardous'].astype(int)

# Remove non-numeric and highly correlated features
df = df.drop(['Orbiting Body', 'Equinox'], axis=1, errors='ignore')

# Plot heatmap to check correlations
plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Drop highly correlated features to avoid redundancy
drop_corr_cols = ['Est Dia in KM(max)', 'Est Dia in M(min)', 'Est Dia in M(max)',
                  'Est Dia in Miles(min)', 'Est Dia in Miles(max)', 'Est Dia in Feet(min)',
                  'Est Dia in Feet(max)', 'Relative Velocity km per hr', 'Miles per hour',
                  'Miss Dist.(lunar)', 'Miss Dist.(kilometers)', 'Miss Dist.(miles)']

df = df.drop(drop_corr_cols, axis=1, errors='ignore')

# Define features (X) and target (y)
target_column = 'Hazardous'
X = df.drop(target_column, axis=1)
y = df[target_column]

# Split dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the XGBoost model
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Feature Importance Plot
plot_importance(xgb_model)
plt.show()

# Make predictions
y_pred = xgb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {np.round(accuracy * 100, 2)}%")

# Define a test dataset with expected features
expected_features = X.columns

# Create a sample dataset for prediction
sample_data = pd.DataFrame([
    [22.5, 0.15, 15.2, 0.05, 1, 0.005, 3.1, 2459000, 0.1, 2.5, 10.5, 80.5, 365.4, 0.2, 150.5, 3.5, 2458800, 12.3, 1.2],
    [18.3, 0.55, 20.8, 0.12, 2, 0.01, 2.8, 2459050, 0.25, 3.1, 15.2, 95.7, 250.7, 0.4, 120.2, 4.2, 2458850, 25.7, 0.8],
    [25.1, 0.10, 8.5, 0.02, 1, 0.002, 3.5, 2459100, 0.08, 2.8, 8.3, 75.8, 415.3, 0.15, 140.3, 3.8, 2458900, 8.9, 1.5],
    [20.7, 0.30, 12.3, 0.09, 3, 0.008, 3.0, 2459150, 0.18, 3.0, 12.1, 110.3, 300.2, 0.3, 110.8, 4.0, 2458950, 17.5, 1.0],
    [17.9, 0.60, 25.0, 0.15, 2, 0.006, 3.2, 2459200, 0.2, 2.9, 14.6, 85.6, 280.6, 0.25, 175.6, 3.9, 2459000, 20.1, 1.3]
], columns=expected_features)

# Predict using the trained model
predictions = xgb_model.predict(sample_data)

# Print results
results = ["Hazardous" if pred == 1 else "Not Hazardous" for pred in predictions]
for i, res in enumerate(results):
    print(f"Asteroid {i+1}: {res}")
