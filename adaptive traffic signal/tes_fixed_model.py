import pickle
import numpy as np
from datetime import datetime

def print_colored(text, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def test_model():
    try:
        # 1. Load the trained model
        with open('traffic_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        print_colored("\n=== Model Loaded Successfully ===", 'green')
        print("Model Type:", type(model).__name__)
        print("Expected Features:", model.feature_names_in_)
        print("Number of Features:", len(model.feature_names_in_))

        # 2. Prepare test input matching ALL required features
        test_input = {
            'Vehicle_Count': 45.0,          # Actual traffic volume
            'Weighted_Vehicle_Count': 54.0, # traffic_volume * 1.2
            'Speed_kmh': 30.0,              # Average speed
            'Density': 22.5,                # traffic_volume / 2
            'Number_of_Lanes': 2.0,         # Default
            'Pedestrian_Crossing': 3.0,     # From your input
            'Bicycle_Lane': 0.0,            # Default no bike lane
            'Hour': datetime.now().hour,    # Current hour
            'DayOfWeek': datetime.now().weekday(), # 0-6
            'Historical_Vehicle_Count': 40.5,# traffic_volume * 0.9
            'Current_Signal_Phase': 0.0      # Default phase
        }

        # 3. Create properly formatted input array
        features = np.array([[test_input[feature] for feature in model.feature_names_in_]])
        
        print_colored("\n=== Test Input ===", 'blue')
        for feature, value in test_input.items():
            print(f"{feature:25}: {value}")

        # 4. Make prediction
        prediction = model.predict(features)
        duration = int(prediction[0])

        print_colored("\n=== Prediction Result ===", 'green')
        print(f"Optimal Green Duration: {duration} seconds")
        
        return True

    except Exception as e:
        print_colored(f"\n!!! Test Failed: {str(e)}", 'red')
        return False

if __name__ == '__main__':
    print_colored("Starting Traffic Model Test...", 'yellow')
    success = test_model()
    
    if success:
        print_colored("\nTest completed successfully!", 'green')
    else:
        print_colored("\nTest failed. Please check the error message.", 'red')