import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('traffic_data.csv')

# Feature Engineering
# Convert categorical variables to numerical
df['Pedestrian_Crossing'] = df['Pedestrian_Crossing'].map({'Yes': 1, 'No': 0})
df['Bicycle_Lane'] = df['Bicycle_Lane'].map({'Yes': 1, 'No': 0})
df['Current_Signal_Phase'] = df['Current_Signal_Phase'].map({'Red': 0, 'Yellow': 1, 'Green': 2})

# Vehicle type weights (trucks need more time to clear intersection)
vehicle_type_weights = {
    'Car': 1.0,
    'Truck': 1.8,
    'Bus': 1.5,
    'Motorcycle': 0.7,
    'Bicycle': 0.5
}
df['Vehicle_Type_Weight'] = df['Vehicle_Type'].map(vehicle_type_weights)

# Time features
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek  # Monday=0, Sunday=6

# Calculate weighted vehicle count
df['Weighted_Vehicle_Count'] = df['Vehicle_Count'] * df['Vehicle_Type_Weight']

# Create target variable (optimal green duration in seconds)
# This is a simplified calculation - in practice you'd want more sophisticated targets
df['Optimal_Green_Duration'] = np.where(
    df['Weighted_Vehicle_Count'] < 10, 20,
    np.where(
        df['Weighted_Vehicle_Count'] < 30, 30,
        np.where(
            df['Weighted_Vehicle_Count'] < 50, 45,
            60  # max 60 seconds
        )
    )
)

# Add pedestrian crossing buffer
df['Optimal_Green_Duration'] += df['Pedestrian_Crossing'] * 5

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
    'Historical_Vehicle_Count'
]

X = df[features]
y = df['Optimal_Green_Duration']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f} seconds")

# Feature importance
importances = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance for Green Light Duration Prediction')
plt.show()

# Example prediction
sample_data = {
    'Vehicle_Count': 25,
    'Weighted_Vehicle_Count': 30,
    'Speed_kmh': 40,
    'Density': 0.8,
    'Number_of_Lanes': 2,
    'Pedestrian_Crossing': 1,
    'Bicycle_Lane': 1,
    'Hour': 17,  # 5 PM
    'DayOfWeek': 2,  # Wednesday
    'Historical_Vehicle_Count': 28
}

sample_df = pd.DataFrame([sample_data])
predicted_duration = model.predict(sample_df)
print(f"\nPredicted optimal green light duration: {predicted_duration[0]:.1f} seconds")