# model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv('traffic_data.csv')

# Data Preprocessing
# Convert categorical variables to numerical
df['Pedestrian_Crossing'] = df['Pedestrian_Crossing'].map({'Yes': 1, 'No': 0})
df['Bicycle_Lane'] = df['Bicycle_Lane'].map({'Yes': 1, 'No': 0})
df['Current_Signal_Phase'] = df['Current_Signal_Phase'].map({'Red': 0, 'Yellow': 1, 'Green': 2})

# Vehicle type weights (trucks need more time to clear intersection)
vehicle_type_weights = {
    'Car': 1.0,
    'Truck': 1.8,  # trucks need more time
    'Bus': 1.5,    # buses need more time
    'Motorcycle': 0.7,  # motorcycles can clear faster
    'Bicycle': 0.5     # bicycles can clear fastest
}
df['Vehicle_Type_Weight'] = df['Vehicle_Type'].map(vehicle_type_weights)

# Time-based features
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek  # Monday=0, Sunday=6

# Calculate weighted vehicle count
df['Weighted_Vehicle_Count'] = df['Vehicle_Count'] * df['Vehicle_Type_Weight']

# Create target variable (optimal green duration in seconds)
# Using a logarithmic scale to account for non-linear relationship
base_duration = 20  # minimum green time
df['Optimal_Green_Duration'] = base_duration + np.log1p(df['Weighted_Vehicle_Count']) * 15

# Add pedestrian crossing buffer
df['Optimal_Green_Duration'] += df['Pedestrian_Crossing'] * 5

# Add density factor (higher density needs longer green)
df['Optimal_Green_Duration'] *= (1 + df['Density'] * 0.3)

# Cap at reasonable maximum (90 seconds)
df['Optimal_Green_Duration'] = np.where(
    df['Optimal_Green_Duration'] > 90,
    90,
    df['Optimal_Green_Duration']
)

# Features for prediction
features = [
    'Vehicle_Count',
    'Weighted_Vehicle_Count',
    'Speed_kmh',
    'Density',
    'Number_of_Lanes',
    'Pedestrian_Crossing',
    'Bicycle_Lane',
    'Hour',
    'DayOfWeek',
    'Historical_Vehicle_Count',
    'Current_Signal_Phase'
]

X = df[features]
y = df['Optimal_Green_Duration']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1  # use all available cores
)

model.fit(X_train, y_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training R²: {train_score:.3f}")
print(f"Testing R²: {test_score:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save the model
with open('traffic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as traffic_model.pkl")

# Optional: Save sample data for testing
sample_data = X_test.iloc[:5].to_dict('records')
with open('sample_data.pkl', 'wb') as f:
    pickle.dump(sample_data, f)
print("Sample test data saved as sample_data.pkl")