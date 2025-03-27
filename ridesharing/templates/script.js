// Initialize impact_stats to prevent "undefined" errors
let impact_stats = {
  congestionLevel: 0,  // Example data (can be replaced with real values)
  averageTravelTime: 0,
  fuelSaved: 0,
};

// Function to update impact statistics
function updateImpactStats() {
  // Example values (Replace with actual data logic)
  impact_stats.congestionLevel = Math.floor(Math.random() * 100); // Simulated congestion % 
  impact_stats.averageTravelTime = Math.floor(Math.random() * 60) + " mins"; // Simulated travel time
  impact_stats.fuelSaved = (Math.random() * 50).toFixed(2) + " liters"; // Simulated fuel savings

  // Update HTML elements with new data
  document.getElementById("congestion").innerText = impact_stats.congestionLevel + "%";
  document.getElementById("travelTime").innerText = impact_stats.averageTravelTime;
  document.getElementById("fuelSaved").innerText = impact_stats.fuelSaved;
}

// Ensure function runs every 5 seconds (simulating real-time updates)
setInterval(updateImpactStats, 5000);

// Run function once at the start to populate values immediately
updateImpactStats();
