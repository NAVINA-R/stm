import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories if they don't exist
os.makedirs('static', exist_ok=True)
os.makedirs('models', exist_ok=True)

def load_data(filename):
    """Load and preprocess the dataset."""
    df = pd.read_csv(filename)
    
    # Convert categorical variables
    day_mapping = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 
        'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    df['day_of_week'] = df['day_of_week'].map(day_mapping)
    
    # Feature engineering
    df['traffic_speed_ratio'] = df['past_hour_avg_speed'] / (df['past_hour_traffic_count'] + 1)
    df['congestion_factor'] = df['traffic_density'] * df['congestion_level']
    
    # Define target variable (1=left, 2=middle, 3=right)
    conditions = [
        (df['congestion_level'] == 0) & (df['past_hour_avg_speed'] > 50),
        (df['congestion_level'] == 1) & (df['traffic_density'] < 100),
        (df['congestion_level'] == 2) | (df['traffic_density'] > 150)
    ]
    choices = [1, 2, 3]
    df['preferred_lane'] = np.select(conditions, choices, default=2)
    
    return df

def train_model(df):
    """Train and evaluate the model."""
    features = [
        'time_of_day', 'day_of_week', 'weather_condition', 'holiday',
        'special_event_nearby', 'past_hour_avg_speed', 'past_hour_traffic_count',
        'road_type', 'accident_nearby', 'road_closure', 'traffic_density',
        'alternative_routes_available', 'congestion_level', 'traffic_speed_ratio',
        'congestion_factor'
    ]
    target = 'preferred_lane'
    
    X = df[features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with good default parameters
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(importance)
    
    # Save model
    joblib.dump(model, 'models/lane_prediction_model.joblib')
    
    return model, importance

def visualize_feature_importance(importance):
    """Generate and save feature importance plot."""
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance)
    plt.title('Feature Importance for Lane Prediction')
    plt.tight_layout()
    plt.savefig('static/feature_importance.png')
    plt.close()

if __name__ == "__main__":
    try:
        print("🚦 Smart Lane Management Model Training 🚦")
        
        # 1. Load and preprocess data
        print("\nLoading data...")
        df = load_data('smart_lane_management_large_dataset.csv')
        
        # 2. Train model
        print("\nTraining model...")
        model, importance = train_model(df)
        
        # 3. Visualize feature importance
        print("\nGenerating visualization...")
        visualize_feature_importance(importance)
        
        print("\n✅ Training complete!")
        print("Model saved to: models/lane_prediction_model.joblib")
        print("Feature importance visualization saved to: static/feature_importance.png")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check:")
        print("1. Input data file exists and is formatted correctly")
        print("2. Required packages are installed")
        print("3. You have write permissions in the directory")