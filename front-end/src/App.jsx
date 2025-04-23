import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar
} from 'recharts';
import './App.css';
import { db } from "./firebase";
import { addDoc, collection } from "firebase/firestore";

function App() {
  const [roads, setRoads] = useState([]);
  const [selectedRoad, setSelectedRoad] = useState(null);
  const [currentTraffic, setCurrentTraffic] = useState({});
  const [predictions, setPredictions] = useState({});
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [chartData, setChartData] = useState([]); // New state for chart data

  // Fetch list of roads and current traffic
  useEffect(() => {
    const fetchRoads = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/traffic/current');
        const data = await response.json();
        


        addDoc(collection(db, "road"), data)
.then(() => {
  console.log("Document successfully written!");
})
.catch((error) => {
  console.error("Error adding document: ", error.code, error.message);
  // Check for specific Firebase errors
  if (error.code === 'permission-denied') {
    console.error("Firebase permission denied - check security rules");
  }
});
        // Extract road IDs from the data
        const roadIds = Object.keys(data.roads);
        setRoads(roadIds);
        
        if (roadIds.length > 0 && !selectedRoad) {
          setSelectedRoad(roadIds[0]);
        }
        
        setCurrentTraffic(data.roads);
        setLoading(false);
        
        // Update chart data with the latest current data
        if (selectedRoad && data.roads[selectedRoad]) {
          updateChartData(data.roads[selectedRoad]);
        }
      } catch (error) {
        console.error('Error fetching roads:', error);
        setLoading(false);
      }
    };

    fetchRoads();
    const interval = setInterval(fetchRoads, 10000); // Refresh every 10 seconds
    
    return () => clearInterval(interval);
  }, [selectedRoad]);

  // Function to update chart data with the latest readings
  const updateChartData = (roadData) => {
    const now = new Date();
    const shortTime = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    const newDataPoint = {
      timestamp: now.toISOString(),
      shortTime,
      congestionLevel: roadData.congestionLevel,
      vehicleCount: roadData.vehicleCount,
      averageSpeed: roadData.averageSpeed
    };
    

    // More robust error handling for Firestore


    setChartData(prevData => {
      // Keep only the last 60 data points (10 minutes if updating every 10 seconds)
      const updatedData = [...prevData, newDataPoint];
      if (updatedData.length > 60) {
        return updatedData.slice(updatedData.length - 60);
      }
      return updatedData;
    });
  };

  // Fetch predictions
  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/traffic/predictions');
        const data = await response.json();
        setPredictions(data.predictions);
      } catch (error) {
        console.error('Error fetching predictions:', error);
      }
    };

    fetchPredictions();
    const interval = setInterval(fetchPredictions, 5000); // Refresh every 5 seconds
    
    return () => clearInterval(interval);
  }, []);

  // Fetch history for selected road and initialize chart data
  useEffect(() => {
    if (!selectedRoad) return;
    
    const fetchHistory = async () => {
      try {
        const response = await fetch(`http://localhost:5000/api/traffic/history/${selectedRoad}?limit=60`);
        const data = await response.json();
        
        // Format timestamps for display
        const formattedHistory = data.history.map(item => ({
          ...item,
          shortTime: new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
        }));
        
        setHistory(formattedHistory);
        
        // Initialize chart data with history when changing roads
        setChartData(formattedHistory);
      } catch (error) {
        console.error(`Error fetching history for road ${selectedRoad}:`, error);
      }
    };

    fetchHistory();
    
    return () => {
      // Clear chart data when changing roads
      setChartData([]);
    };
  }, [selectedRoad]);

  // Get congestion level text and color
  const getCongestionInfo = (level) => {
    const levels = [
      { text: 'Free Flow', color: '#4CAF50' },
      { text: 'Light', color: '#8BC34A' },
      { text: 'Moderate', color: '#FFEB3B' },
      { text: 'Heavy', color: '#FF9800' },
      { text: 'Very Heavy', color: '#F44336' },
      { text: 'Gridlock', color: '#B71C1C' }
    ];
    
    return levels[level] || { text: 'Unknown', color: '#9E9E9E' };
  };

  // Handle road selection change
  const handleRoadChange = (e) => {
    const newRoad = e.target.value;
    setSelectedRoad(newRoad);
    // When changing roads, immediately update chart with current data
    if (currentTraffic[newRoad]) {
      updateChartData(currentTraffic[newRoad]);
    }
  };

  if (loading) {
    return (
      <div className="loading">
        <h2>Loading Tariq Traffic Dashboard...</h2>
      </div>
    );
  }

  // Prepare current road data
  const currentRoadData = selectedRoad && currentTraffic[selectedRoad] ? currentTraffic[selectedRoad] : null;
  const currentRoadPrediction = selectedRoad && predictions[selectedRoad] ? predictions[selectedRoad] : null;
  
  // Use merged data for charts (combines history with current updates)
  const displayData = chartData.length > 0 ? chartData : history;
  
  return (
    <div className="app">
      <header className="header">
        <h1>🚦 Tariq Traffic Management Dashboard</h1>
        <div className="road-selector">
          <label>Select Road: </label>
          <select 
            value={selectedRoad || ''} 
            onChange={handleRoadChange}
          >
            {roads.map(roadId => (
              <option key={roadId} value={roadId}>
                {currentTraffic[roadId]?.roadName || roadId}
              </option>
            ))}
          </select>
        </div>
      </header>

      <div className="dashboard">
        <div className="cards">
          {currentRoadData && (
            <div className="card current-status">
              <h2>Current Status</h2>
              <div className="status-indicators">
                <div className="indicator">
                  <span className="label">Vehicles:</span>
                  <span className="value">{currentRoadData.vehicleCount}</span>
                </div>
                
                <div className="indicator">
                  <span className="label">Average Speed:</span>
                  <span className="value">{currentRoadData.averageSpeed.toFixed(1)} km/h</span>
                </div>
                
                <div className="indicator">
                  <span className="label">Congestion:</span>
                  <span 
                    className="value congestion-level"
                    style={{ backgroundColor: getCongestionInfo(currentRoadData.congestionLevel).color }}
                  >
                    {getCongestionInfo(currentRoadData.congestionLevel).text}
                  </span>
                </div>
              </div>
            </div>
          )}
          
          {currentRoadPrediction && (
            <div className="card prediction">
              <h2>Prediction</h2>
              <div className="prediction-container">
                <div className="prediction-label">
                  Predicted Congestion:
                </div>
                <div 
                  className="prediction-value"
                  style={{ backgroundColor: getCongestionInfo(currentRoadPrediction.predicted_congestion).color }}
                >
                  {getCongestionInfo(currentRoadPrediction.predicted_congestion).text}
                </div>
              </div>
            </div>
          )}
        </div>
        
        <div className="charts">
          <div className="chart-container">
            <h2>Congestion History</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={displayData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="shortTime" 
                  tick={{ fontSize: 12 }}
                  interval="preserveStartEnd"
                />
                <YAxis domain={[0, 5]} />
                <Tooltip 
                  formatter={(value, name) => {
                    if (name === "Congestion Level") {
                      return [getCongestionInfo(value).text, name];
                    }
                    return [value, name];
                  }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="congestionLevel" 
                  name="Congestion Level" 
                  stroke="#8884d8" 
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 8 }} 
                  isAnimationActive={true}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          <div className="chart-container">
            <h2>Vehicle Count History</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={displayData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="shortTime" 
                  tick={{ fontSize: 12 }}
                  interval="preserveStartEnd"
                />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar 
                  dataKey="vehicleCount" 
                  name="Vehicle Count" 
                  fill="#82ca9d"
                  isAnimationActive={true}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;