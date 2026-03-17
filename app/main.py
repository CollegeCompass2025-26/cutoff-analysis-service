from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="CollegeCompass Cutoff Analysis Service",
    version="1.0.0",
    description="Microservice for PDF data extraction and ML-based cutoff trend analysis."
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.services.trends.predictor import TrendPredictor

# Initialize predictor
predictor = TrendPredictor()

class PredictionRequest(BaseModel):
    counseling_source: str # e.g., 'JoSAA', 'MCC'
    exam_type: str # e.g., 'JEE-ADV', 'NEET-UG'
    college_name: str
    course_name: str
    category: str
    user_rank: int
    user_location: Optional[str] = None # e.g., 'Mumbai, Maharashtra'
    history: Optional[List[int]] = None # Previous years ranks

class RoundDrift(BaseModel):
    round_name: str
    delta: int # Rank inflation/deflation relative to Round 1

class Competitor(BaseModel):
    college_name: str
    avg_rank: int
    similarity_score: float

class PredictionResponse(BaseModel):
    # 1. Predicted Ranges
    predicted_rank: int
    
    # 2. Trend Forecasting
    trend_tag: str # UPWARD, DOWNWARD, STABLE
    
    # 3. Anomaly Detection
    is_anomaly: bool
    anomaly_score: float
    
    # 4. Admission Probability
    admission_probability: float
    
    # 5. Volatility Index
    volatility_score: float
    
    # 6. Round-wise Drift
    round_drift: List[RoundDrift]
    
    # 7. Multi-College Benchmarking
    competitors: List[Competitor]
    
    # 8. Temporal Strategy
    recommended_round: str
    strategy_insights: List[str] # Multiple strategic recommendations
    
    # 9. AI Insights (SHAP based)
    insights: List[str]
    
    # 10. Regional Heatmap (Geo-Spatial)
    region_competition_index: float # 0-100 scale for that state/city
    user_location_context: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "cutoff-analysis-service", "timestamp": time.time()}

@app.post("/api/v1/trends/predict", response_model=PredictionResponse)
async def predict_cutoff(request: PredictionRequest):
    try:
        analysis = predictor.get_full_analysis(
            source=request.counseling_source,
            exam_type=request.exam_type,
            college_name=request.college_name,
            course_name=request.course_name,
            category=request.category,
            user_rank=request.user_rank,
            user_location=request.user_location,
            history=request.history
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/trends/models")
async def get_models():
    return {
        "active_models": ["XGBoost", "LSTM", "FNN", "CNN", "RandomForest"],
        "ensembling_enabled": True
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
