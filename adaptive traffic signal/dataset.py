import pandas as pd
import random
from datetime import datetime, timedelta

# Define parameters
latitude = 8.76735
longitude = 78.13425
num_records = 500  # Number of data points to generate

# Possible categories
vehicle_types = ["Car", "Truck", "Bus", "Motorcycle", "Bicycle"]
directions = ["North", "South", "East", "West"]
lanes = [1, 2, 3, 4]
turn_movements = ["Left-Turn", "Right-Turn", "Through"]
weather_conditions = ["Clear", "Rain", "Fog", "Snow"]
signal_phases = {"Green": 30, "Yellow": 5, "Red": 45}  # Average durations

def generate_random_time():
    base_time = datetime(2024, 1, 1, 0, 0)  # Starting point
    return base_time + timedelta(minutes=random.randint(0, 365 * 24 * 60))

def generate_random_data():
    return {
        "Timestamp": generate_random_time().strftime('%Y-%m-%d %H:%M:%S'),
        "Latitude": latitude,
        "Longitude": longitude,
        "Vehicle_Count": random.randint(1, 50),
        "Vehicle_Type": random.choice(vehicle_types),
        "Direction": random.choice(directions),
        "Speed_kmh": random.uniform(10, 80),
        "Density": random.uniform(0.1, 1.5),
        "Number_of_Lanes": random.choice(lanes),
        "Turn_Movement": random.choice(turn_movements),
        "Pedestrian_Crossing": random.choice(["Yes", "No"]),
        "Bicycle_Lane": random.choice(["Yes", "No"]),
        "Current_Signal_Phase": random.choice(list(signal_phases.keys())),
        "Cycle_Length": sum(signal_phases.values()),
        "Phase_Split": random.choice(list(signal_phases.values())),
        "Offset": random.randint(0, 30),
        "Time_of_Day": random.choice(["Morning", "Afternoon", "Evening", "Night"]),
        "Day_of_Week": random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
        "Weather_Conditions": random.choice(weather_conditions),
        "Special_Event": random.choice(["Yes", "No"]),
        "Historical_Vehicle_Count": random.randint(5, 100),
        "Historical_Signal_Timing": random.choice(list(signal_phases.values())),
        "Incident_Data": random.choice(["Accident", "Road Closure", "None"])
    }

# Generate dataset
data = [generate_random_data() for _ in range(num_records)]
df = pd.DataFrame(data)

# Save to CSV file
df.to_csv("traffic_data.csv", index=False)
print("Dataset saved as traffic_data.csv")
