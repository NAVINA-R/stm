from flask import Flask, render_template
import ride_sharing_optimizer as rso
import os
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def home():
    try:
        # Run analysis
        df = rso.load_and_preprocess_data('ride_sharing_dataset.csv')
        demand_results = rso.predict_high_demand_zones(df)
        carpool_matches = rso.find_carpool_matches(df)
        
        # Generate unique filenames for plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('static/images', exist_ok=True)
        
        hotspot_plot = f"images/hotspots_{timestamp}.png"
        hourly_plot = f"images/hourly_{timestamp}.png"
        
        # Save plots
        rso.visualize_results(demand_results, carpool_matches, 
                            save_paths={
                                'hotspot_plot': f'static/{hotspot_plot}',
                                'hourly_plot': f'static/{hourly_plot}'
                            })
        
        # Prepare data for template
        top_matches = carpool_matches.head(10).to_dict('records')
        top_hotspots = demand_results['hotspot_counts'].head(5).to_dict()
        
        return render_template('index.html', 
                            hotspot_plot=hotspot_plot,
                            hourly_plot=hourly_plot,
                            top_matches=top_matches,
                            top_hotspots=top_hotspots)
    
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)