import requests
import json
import time

BASE_URL = "http://localhost:8001"

def run_edge_test(name, payload):
    print(f"\n--- Testing Edge Case: {name} ---")
    try:
        start = time.time()
        resp = requests.post(f"{BASE_URL}/api/v1/trends/predict", json=payload)
        latency = time.time() - start
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ Success! (Latency: {latency:.2f}s)")
            print(f"   Prediction: {data['predicted_rank']}")
            print(f"   Anomaly Detected: {data['is_anomaly']}")
            print(f"   Volatility Score: {data['volatility_score']}")
            print(f"   Strategy Insights: {len(data['strategy_insights'])} returned")
        else:
            print(f"❌ Failed (Status {resp.status_code}):")
            print(f"   {resp.text}")
    except Exception as e:
        print(f"🧨 Exception: {e}")

# 1. New Entity (Empty History)
run_edge_test("Empty History (Cold Start)", {
    "counseling_source": "JoSAA",
    "exam_type": "JEE-ADV",
    "college_name": "New Institute",
    "course_name": "New Engineering Dept",
    "category": "OPEN",
    "user_rank": 1000,
    "history": [] # No historical data
})

# 2. Extreme Volatility
run_edge_test("High Volatility History", {
    "counseling_source": "JoSAA",
    "exam_type": "JEE-ADV",
    "college_name": "Volatile College",
    "course_name": "CS",
    "category": "OBC-NCL",
    "user_rank": 500,
    "history": [100, 5000, 50, 10000] # Random jumps
})

# 3. Rank Near Zero (Small value validation)
run_edge_test("Pioneer Rank (Rank 1)", {
    "counseling_source": "MCC",
    "exam_type": "NEET-UG",
    "college_name": "AIIMS Delhi",
    "course_name": "MBBS",
    "category": "GENERAL",
    "user_rank": 1, 
    "history": [1, 2, 1, 3]
})

# 4. Outlier Prediction (Testing Anomaly Trigger)
run_edge_test("Anomaly Trigger (Huge Jump)", {
    "counseling_source": "JoSAA",
    "exam_type": "JEE-ADV",
    "college_name": "Stable College",
    "course_name": "Civil",
    "category": "SC",
    "user_rank": 40000,
    "history": [40000, 41000, 40500] 
    # Logic in predictor: dev > 0.5 triggers anomaly. 
    # If model predicts 100k here, it should trigger.
})

# 5. Missing Location
run_edge_test("Missing User Location", {
    "counseling_source": "MHT-CET",
    "exam_type": "CET",
    "college_name": "COEP Pune",
    "course_name": "IT",
    "category": "OPEN",
    "user_rank": 1200,
    "user_location": None # Null check
})

# 6. Invalid Counseling Source (Resilience)
run_edge_test("Unknown Source Context", {
    "counseling_source": "UNKNOWN_SITE_XYZ",
    "exam_type": "FANTASY_EXAM",
    "college_name": "Hogwarts",
    "course_name": "Magic",
    "category": "OPEN",
    "user_rank": 500,
    "history": [400, 450]
})

# 7. Zero History Value (Log Transform Crash Check)
run_edge_test("Zero Value in History", {
    "counseling_source": "JoSAA",
    "exam_type": "JEE-ADV",
    "college_name": "Test College",
    "course_name": "Testing",
    "category": "OPEN",
    "user_rank": 500,
    "history": [0, 500] # 0 can crash np.log
})
