from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the trained model
model = joblib.load('lane_prediction_model.joblib')

# Mapping for day of week
day_mapping = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert to numerical values
        time_of_day = int(data['time_of_day'])
        day_of_week = day_mapping[data['day_of_week']]
        weather_condition = int(data['weather_condition'])
        holiday = 1 if 'holiday' in data else 0
        special_event = 1 if 'special_event' in data else 0
        avg_speed = float(data['avg_speed'])
        traffic_count = int(data['traffic_count'])
        road_type = int(data['road_type'])
        accident = 1 if 'accident' in data else 0
        road_closure = 1 if 'road_closure' in data else 0
        traffic_density = int(data['traffic_density'])
        alt_routes = 1 if 'alt_routes' in data else 0
        congestion_level = int(data['congestion_level'])
        
        # Calculate derived features
        traffic_speed_ratio = avg_speed / (traffic_count + 1)
        congestion_factor = traffic_density * congestion_level
        
        # Create feature array
        features = [
            time_of_day, day_of_week, weather_condition, holiday,
            special_event, avg_speed, traffic_count, road_type,
            accident, road_closure, traffic_density, alt_routes,
            congestion_level, traffic_speed_ratio, congestion_factor
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Map prediction to lane name
        lane_mapping = {1: 'Left Lane', 2: 'Middle Lane', 3: 'Right Lane'}
        recommended_lane = lane_mapping.get(prediction, 'Middle Lane')
        
        # Get current time for the response
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'recommended_lane': recommended_lane,
            'timestamp': current_time
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)