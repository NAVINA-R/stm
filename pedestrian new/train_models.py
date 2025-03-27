# train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('pedestrian_traffic_data.csv')

# Data Preprocessing
# Convert timestamp to datetime and extract features
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Hour'] = data['Timestamp'].dt.hour
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
data['Month'] = data['Timestamp'].dt.month

# Handle missing values if any
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
categorical_cols = ['Behavior', 'Age_Group', 'Peak_OffPeak', 'Vehicle_Type', 
                    'Weather', 'Lighting_Condition', 'Road_Surface', 
                    'Traffic_Signal_Phase', 'Pedestrian_Signal', 'Event_Type',
                    'Congestion_Level']  # Added Congestion_Level

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Save label encoders for later use
joblib.dump(label_encoders, 'label_encoders.pkl')

# Feature selection for crossing pattern prediction
crossing_features = ['Hour', 'DayOfWeek', 'Month', 'Walking_Speed_mps', 
                    'Density_people_per_m2', 'Behavior', 'Age_Group', 
                    'Peak_OffPeak', 'Vehicle_Count_per_min', 
                    'Pedestrian_Count_per_min', 'Congestion_Level', 
                    'Weather', 'Temperature_C', 'Lighting_Condition', 
                    'Road_Surface', 'Traffic_Signal_Phase', 'Pedestrian_Signal']

X_cross = data[crossing_features]
y_cross = data['Behavior']  # Predicting crossing behavior

# Split data for crossing pattern prediction
X_train_cross, X_test_cross, y_train_cross, y_test_cross = train_test_split(
    X_cross, y_cross, test_size=0.2, random_state=42)

# Train Random Forest Classifier for crossing patterns
cross_model = RandomForestClassifier(n_estimators=100, random_state=42)
cross_model.fit(X_train_cross, y_train_cross)

# Evaluate crossing pattern model
cross_pred = cross_model.predict(X_test_cross)
print("Crossing Pattern Prediction Report:")
print(classification_report(y_test_cross, cross_pred))

# Save crossing pattern model
joblib.dump(cross_model, 'crossing_pattern_model.pkl')

# Feature selection for wait time prediction
# In the feature selection for wait time prediction section
wait_time_features = ['Hour', 'DayOfWeek', 'Month', 'Density_people_per_m2', 
                     'Pedestrian_Count_per_min', 'Congestion_Level', 
                     'Vehicle_Count_per_min', 
                     'Traffic_Signal_Phase', 'Pedestrian_Signal', 
                     'Weather', 'Temperature_C', 'Lighting_Condition']
# Removed 'Signal_Duration_s' since it's the target variable
X_wait = data[wait_time_features]
y_wait = data['Signal_Duration_s']  # Predicting signal duration (proxy for wait time)

# Split data for wait time prediction
X_train_wait, X_test_wait, y_train_wait, y_test_wait = train_test_split(
    X_wait, y_wait, test_size=0.2, random_state=42)

# Scale features for wait time prediction
scaler = StandardScaler()
X_train_wait_scaled = scaler.fit_transform(X_train_wait)
X_test_wait_scaled = scaler.transform(X_test_wait)

# Train Random Forest Regressor for wait time
wait_model = RandomForestRegressor(n_estimators=100, random_state=42)
wait_model.fit(X_train_wait_scaled, y_train_wait)

# Evaluate wait time model
wait_pred = wait_model.predict(X_test_wait_scaled)
mse = mean_squared_error(y_test_wait, wait_pred)
print(f"\nWait Time Prediction MSE: {mse:.2f}")
print(f"RMSE: {np.sqrt(mse):.2f}")

# Save wait time model and scaler
joblib.dump(wait_model, 'wait_time_model.pkl')
joblib.dump(scaler, 'wait_time_scaler.pkl')

print("\nModels trained and saved successfully!")