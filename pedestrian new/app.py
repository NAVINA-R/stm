# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

app = Flask(__name__)

# Load models and encoders
cross_model = joblib.load('crossing_pattern_model.pkl')
wait_model = joblib.load('wait_time_model.pkl')
scaler = joblib.load('wait_time_scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()
        
        # Convert timestamp to features
        timestamp = datetime.strptime(data['timestamp'], '%Y-%m-%dT%H:%M')
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        
        # Prepare crossing pattern prediction features
        crossing_features = {
            'Hour': hour,
            'DayOfWeek': day_of_week,
            'Month': month,
            'Walking_Speed_mps': float(data['walking_speed']),
            'Density_people_per_m2': float(data['density']),
            'Behavior': data['behavior'],
            'Age_Group': data['age_group'],
            'Peak_OffPeak': data['peak_offpeak'],
            'Vehicle_Count_per_min': int(data['vehicle_count']),
            'Pedestrian_Count_per_min': int(data['pedestrian_count']),
            'Congestion_Level': data['congestion_level'],
            'Weather': data['weather'],
            'Temperature_C': float(data['temperature']),
            'Lighting_Condition': data['lighting_condition'],
            'Road_Surface': data['road_surface'],
            'Traffic_Signal_Phase': data['traffic_signal_phase'],
            'Pedestrian_Signal': data['pedestrian_signal']
        }
        
        # Encode categorical features for crossing pattern
        crossing_df = pd.DataFrame([crossing_features])
        for col in crossing_df.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                crossing_df[col] = label_encoders[col].transform(crossing_df[col])
        
        # Predict crossing pattern
        crossing_pred = cross_model.predict(crossing_df)[0]
        crossing_prob = cross_model.predict_proba(crossing_df)[0]
        
        # Get behavior label back from encoder
        behavior_labels = label_encoders['Behavior'].classes_
        predicted_behavior = behavior_labels[crossing_pred]
        
        # Prepare wait time prediction features
        wait_time_features = {
            'Hour': hour,
            'DayOfWeek': day_of_week,
            'Month': month,
            'Density_people_per_m2': float(data['density']),
            'Pedestrian_Count_per_min': int(data['pedestrian_count']),
            'Congestion_Level': data['congestion_level'],
            'Vehicle_Count_per_min': int(data['vehicle_count']),
             # default value, will be overwritten
            'Traffic_Signal_Phase': data['traffic_signal_phase'],
            'Pedestrian_Signal': data['pedestrian_signal'],
            'Weather': data['weather'],
            'Temperature_C': float(data['temperature']),
            'Lighting_Condition': data['lighting_condition']
        }
        
        # Encode categorical features for wait time
        wait_df = pd.DataFrame([wait_time_features])
        for col in wait_df.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                wait_df[col] = label_encoders[col].transform(wait_df[col])
        
        # Scale features and predict wait time
        wait_df_scaled = scaler.transform(wait_df)
        wait_time_pred = wait_model.predict(wait_df_scaled)[0]
        
        # Prepare response
        response = {
            'predicted_behavior': predicted_behavior,
            'behavior_probabilities': {label: float(prob) for label, prob in zip(behavior_labels, crossing_prob)},
            'predicted_wait_time': round(float(wait_time_pred), 1),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M')
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)