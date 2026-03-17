import requests
import time

def test_api():
    base_url = "http://localhost:8001"
    
    # 1. Health Check
    print("Testing Health Check...")
    try:
        resp = requests.get(f"{base_url}/health")
        print(f"Health Resp: {resp.json()}")
    except Exception as e:
        print(f"Health Check Failed: {e}")
        return

    # 2. Model List
    print("\nTesting Model List...")
    resp = requests.get(f"{base_url}/api/v1/trends/models")
    print(f"Models: {resp.json()}")

    # 3. Predict Cutoff (Full 10-Feature Analysis with Location)
    print("\nTesting Full 10-Feature Predict Analysis...")
    payload = {
        "counseling_source": "JoSAA",
        "exam_type": "JEE-ADV",
        "college_name": "Indian Institute of Technology Bombay",
        "course_name": "Computer Science and Engineering (4 Years, Bachelor of Technology)",
        "category": "OPEN",
        "user_rank": 250,
        "user_location": "Mumbai, Maharashtra",
        "history": [190, 210, 205] 
    }
    
    start_time = time.time()
    resp = requests.post(f"{base_url}/api/v1/trends/predict", json=payload)
    end_time = time.time()
    
    if resp.status_code == 200:
        data = resp.json()
        print(f"10-Feature Analysis (Latency: {end_time - start_time:.2f}s):")
        print(f"  A1. Predicted Rank: {data['predicted_rank']}")
        print(f"  A2. Trend: {data['trend_tag']}")
        print(f"  A3. Anomaly: {data['is_anomaly']} (Score: {data['anomaly_score']})")
        print(f"  A4. Probability: {data['admission_probability']}%")
        print(f"  A5. Volatility: {data['volatility_score']}")
        print(f"  A6. Drift (Delta R3/R1): {data['round_drift'][-1]['delta']}")
        print(f"  A8. Strategies: {data['strategy_insights']}")
        print(f"  A10. Location Context: {data['user_location_context']}")
        print(f"  A10. Map Coords: {data['coordinates']}")
    else:
        print(f"Analysis Failed (Status {resp.status_code}): {resp.text}")

if __name__ == "__main__":
    test_api()
