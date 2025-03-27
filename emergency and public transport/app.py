from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

app = Flask(__name__)

# Load your pre-trained models
try:
    route_model = joblib.load('route_model.pkl')
    delay_model = joblib.load('delay_model.pkl')
except Exception as e:
    print(f"Error loading models: {e}")
    route_model = None
    delay_model = None

# Sample route data for Thoothukudi
routes = {
    'Route1': {
        'name': 'Harbor to General Hospital',
        'distance': 3.5, 
        'traffic_lights': 2,
        'hospitals': 1,
        'coordinates': [
            [8.764166, 78.134834],  # Harbor area
            [8.768912, 78.138745],  # Main road
            [8.772345, 78.142567]   # General Hospital
        ]
    },
    'Route2': {
        'name': 'Railway Station to SPIC Hospital',
        'distance': 5.2,
        'traffic_lights': 3,
        'hospitals': 1,
        'coordinates': [
            [8.728945, 78.126734],  # Railway Station
            [8.735672, 78.130845],
            [8.742389, 78.135956],
            [8.746782, 78.140067]    # SPIC Hospital
        ]
    },
    'Route3': {
        'name': 'Bus Stand to Thoothukudi Medical College',
        'distance': 4.8,
        'traffic_lights': 4,
        'hospitals': 1,
        'coordinates': [
            [8.752367, 78.122345],  # Bus Stand
            [8.756782, 78.126456],
            [8.761194, 78.130567],
            [8.765607, 78.134678],
            [8.770020, 78.138789]    # Medical College
        ]
    }
}

@app.route('/')
def index():
    return render_template('index.html', routes=routes)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features for prediction
        vehicle_type = data['vehicle_type']
        start_point = data['start_point']
        end_point = data['end_point']
        time_of_day = data['time_of_day']
        day_of_week = data['day_of_week']
        
        # Prepare features for models
        features = {
            'vehicle_type': 1 if vehicle_type == 'ambulance' else 0,
            'distance': routes[end_point]['distance'] - routes[start_point]['distance'],
            'traffic_lights': routes[end_point]['traffic_lights'],
            'time_of_day': time_of_day,
            'day_of_week': day_of_week,
            'is_peak_hour': 1 if (7 <= int(time_of_day.split(':')[0]) <= 10 or 16 <= int(time_of_day.split(':')[0]) <= 19) else 0
        }
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Predict best route
        if route_model:
            route_prediction = route_model.predict(features_df)[0]
        else:
            route_prediction = "Route2"  # fallback
            
        # Predict delays
        if delay_model:
            delay_prediction = delay_model.predict(features_df)[0]
        else:
            delay_prediction = 5  # fallback in minutes
            
        # Optimize traffic light preemption
        traffic_lights = optimize_traffic_lights(route_prediction, delay_prediction, vehicle_type)
        
        return jsonify({
            'status': 'success',
            'best_route': route_prediction,
            'route_name': routes[route_prediction]['name'],
            'predicted_delay': delay_prediction,
            'traffic_light_adjustments': traffic_lights,
            'route_coordinates': routes[route_prediction]['coordinates'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def optimize_traffic_lights(route, delay, vehicle_type):
    """Optimize traffic light timing based on predictions"""
    adjustments = {}
    
    # Thoothukudi-specific traffic light names
    if route == 'Route1':
        adjustments = {
            'Harbor Signal': {'action': 'extend_green', 'duration': min(30, delay * 8)},
            'Collectorate Signal': {'action': 'early_green', 'duration': 12}
        }
    elif route == 'Route2':
        adjustments = {
            'Railway Signal': {'action': 'extend_green', 'duration': min(25, delay * 7)},
            'VOC Signal': {'action': 'hold_green', 'duration': 15}
        }
    else:
        adjustments = {
            'Bus Stand Signal': {'action': 'early_green', 'duration': 10},
            'Medical College Signal': {'action': 'extend_green', 'duration': min(20, delay * 6)}
        }
    
    return adjustments

if __name__ == '__main__':
    app.run(debug=True)