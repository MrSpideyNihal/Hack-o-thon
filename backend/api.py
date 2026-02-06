from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
from drift_detector import DriftDetector
import json

app = FastAPI(title="DriftGuard API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = DriftDetector(
    baseline_window_days=7,
    alert_threshold_strikes=3
)

# Load baseline
baseline_df = pd.read_csv('../data/baseline_data.csv')

@app.get("/")
def root():
    return {
        "service": "DriftGuard API",
        "status": "online",
        "version": "1.0.0"
    }

@app.post("/detect-drift")
def detect_drift(scenario: str = "A"):
    """
    Detect drift in specified scenario
    """
    try:
        # Load drift scenario
        scenario_file = f'../data/drift_scenario_{scenario}.csv'
        current_df = pd.read_csv(scenario_file)
        
        # Run detection
        features = ['income', 'credit_score', 'age', 'debt_ratio']
        results = detector.detect_drift(baseline_df, current_df, features)
        
        # Convert to JSON-serializable format
        return {
            "timestamp": results['timestamp'].isoformat(),
            "drift_detected": results['overall_drift'],
            "features": results['features'],
            "alerts": results['alerts'],
            "subpopulation_drift": results['subpopulation_drift'][:5],  # Top 5
            "model_confidence": results.get('model_confidence', 0.0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift-status")
def get_drift_status():
    """
    Get current drift monitoring status
    """
    summary = detector.get_drift_summary()
    return summary

@app.get("/drift-history")
def get_drift_history():
    """
    Get historical drift data for dashboard
    """
    if not detector.drift_history:
        return []
    
    return [
        {
            "timestamp": h['timestamp'].isoformat(),
            "drift_detected": h['overall_drift'],
            "alert_count": len(h['alerts'])
        }
        for h in detector.drift_history[-50:]  # Last 50 checks
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
