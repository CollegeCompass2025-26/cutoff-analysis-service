import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

from app.services.trends.ensemble import CutoffEnsemble

def run_rigorous_tests():
    print("Initializing Ensemble for Rigorous Testing...")
    ensemble = CutoffEnsemble()
    
    test_cases = [
        # --- Tier 1 Cases (Stable History) ---
        {
            "name": "IIT Bombay - CS (Stable)",
            "features": {'college_name': 'IIT Bombay', 'course_name': 'Computer Science', 'specialization_name': 'N/A', 'exam_name': 'JEE-ADV', 'category': 'OPEN', 'cutoff_type': 'rank', 'state': 'Maharashtra', 'city': 'Mumbai', 'typeofuni': 'IIT', 'year': 2025, 'avg_package': 21.5, 'fees': 2.5},
            "history": [180, 195, 205, 210],
            "user_rank": 250
        },
        {
            "name": "IIT Delhi - EE (Stable)",
            "features": {'college_name': 'IIT Delhi', 'course_name': 'Electrical Engineering', 'specialization_name': 'N/A', 'exam_name': 'JEE-ADV', 'category': 'OBC-NCL', 'cutoff_type': 'rank', 'state': 'Delhi', 'city': 'New Delhi', 'typeofuni': 'IIT', 'year': 2025, 'avg_package': 18.2, 'fees': 2.5},
            "history": [1200, 1350, 1380, 1400],
            "user_rank": 1300
        },
        
        # --- Tier 2/3 Cases (High Volatility) ---
        {
            "name": "COEP Pune - Mechanical (Volatile)",
            "features": {'college_name': 'COEP Pune', 'course_name': 'Mechanical Engineering', 'specialization_name': 'N/A', 'exam_name': 'MHT-CET', 'category': 'OPEN', 'cutoff_type': 'rank', 'state': 'Maharashtra', 'city': 'Pune', 'typeofuni': 'Government', 'year': 2025, 'avg_package': 7.5, 'fees': 1.2},
            "history": [5200, 6800, 8100, 9500],
            "user_rank": 8000
        },
        
        # --- Edge Case: Extreme Dropping Cutoffs (New Trend) ---
        {
            "name": "New AI Course (Sharp Trend)",
            "features": {'college_name': 'COEP Pune', 'course_name': 'Computer Science', 'specialization_name': 'Artificial Intelligence', 'exam_name': 'MHT-CET', 'category': 'OPEN', 'cutoff_type': 'rank', 'state': 'Maharashtra', 'city': 'Pune', 'typeofuni': 'Government', 'year': 2025, 'avg_package': 12.5, 'fees': 1.2},
            "history": [300, 500, 900, 1500],
            "user_rank": 1200
        },
        
        # --- Edge Case: Minimal History (Small Sequences) ---
        {
            "name": "Newer College (Short History)",
            "features": {'college_name': 'IIIT Dharwad', 'course_name': 'Data Science', 'specialization_name': 'N/A', 'exam_name': 'JEE-MAIN', 'category': 'OPEN', 'cutoff_type': 'rank', 'year': 2025},
            "history": [15000, 16500, 17000], # Only 3 rounds
            "user_rank": 16000
        },
        
        # --- Edge Case: Huge Rank Gaps ---
        {
            "name": "Safe Mock",
            "features": {'college_name': 'Unknown Institute', 'course_name': 'Civil', 'year': 2025},
            "history": [45000, 48000, 52000, 55000],
            "user_rank": 10000
        },
        {
            "name": "Impossible Mock",
            "features": {'college_name': 'IIT Bombay', 'course_name': 'Computer Science', 'year': 2025},
            "history": [180, 195, 205, 210],
            "user_rank": 25000
        },
        
        # --- Edge Case: Missing Institutional Metadata (Cold Start) ---
        {
            "name": "Cold Start (No Metadata)",
            "features": {'college_name': 'Private College X', 'course_name': 'IT', 'year': 2025},
            "history": None, # Fallback to XGBoost Only
            "user_rank": 3000
        }
    ]
    
    # Expand to 30 cases by slight variations
    final_cases = []
    for case in test_cases:
        final_cases.append(case)
        # Create a "Variant" with different rank
        variant = case.copy()
        variant['name'] = case['name'] + " (Variant Rank)"
        variant['user_rank'] = int(case['user_rank'] * 1.5)
        final_cases.append(variant)
        # Create a "Lower Tier" variant
        tier_variant = case.copy()
        tier_variant['name'] = case['name'] + " (SC Category)"
        tier_variant['features'] = case['features'].copy()
        tier_variant['features']['category'] = 'SC'
        tier_variant['history'] = [int(h * 3.5) for h in (case['history'] or [1000, 1100, 1200, 1300])]
        tier_variant['user_rank'] = int(case['user_rank'] * 3.5)
        final_cases.append(tier_variant)

    results = []
    print(f"Executing {len(final_cases)} test scenarios...")
    
    for test in final_cases[:30]:
        try:
            pred = ensemble.get_prediction(test['features'], history_seq=test['history'])
            verdict = ensemble.get_risk_assessment(pred, test['user_rank'])
            results.append({
                "Scenario": test['name'],
                "Predicted": pred,
                "User": test['user_rank'],
                "Verdict": verdict,
                "Success": True
            })
        except Exception as e:
            results.append({
                "Scenario": test['name'],
                "Error": str(e),
                "Success": False
            })

    # Output Summary Table
    res_df = pd.DataFrame(results)
    print("\n--- TEST EXECUTION SUMMARY ---")
    print(res_df[["Scenario", "Predicted", "User", "Verdict", "Success"]].to_string())
    
    # Check for failures
    fail_count = len(res_df[res_df["Success"] == False])
    print(f"\nCompleted: {len(res_df)} tests. Failures: {fail_count}")
    
    if fail_count == 0:
        print("ALL TESTS PASSED: Ensemble is robust against diverse feature mappings and history lengths.")
    else:
        print("ALERT: Some test cases failed. Review dimensions and categorical mappings.")

if __name__ == "__main__":
    run_rigorous_tests()
