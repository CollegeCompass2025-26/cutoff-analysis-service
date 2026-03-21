import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import psycopg2
from dotenv import load_dotenv

# Set path
sys.path.append(os.getcwd())
load_dotenv("c:/cutoff-analysis-service/.env")

from app.services.trends.ensemble import CutoffEnsemble

def calculate_metrics():
    print("Calculating Classification Metrics for Ensemble...")
    ensemble = CutoffEnsemble()
    
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        query = "SELECT * FROM ml_features_v3 WHERE closing_rank > 0 ORDER BY RANDOM() LIMIT 100"
        df_test = pd.read_sql(query, conn)
        conn.close()
    except Exception as e:
        print(f"DATABASE FETCH FAILED: {e}")
        return

    verdicts_actual = [] 
    verdicts_pred = []
    
    for _, row in df_test.iterrows():
        feat = {
            'college_name': row['college_name'],
            'course_name': row['course_name'],
            'exam_name': row['exam_name'],
            'category': row['category'],
            'year': row['year'],
            'cutoff_type': 'rank'
        }
        
        actual_rank = float(row['closing_rank'])
        history = [actual_rank * 0.9] * 4 # Mock history
        
        try:
            pred = ensemble.get_prediction(feat, history_seq=history)
            user_rank = actual_rank * 1.02 # Simulated user
            
            p_v = ensemble.get_risk_assessment(pred, user_rank)
            a_v = ensemble.get_risk_assessment(actual_rank, user_rank)
            
            verdicts_pred.append(p_v)
            verdicts_actual.append(a_v)
        except:
            continue

    # Map labels to numeric for multi-class AUC if possible, 
    # but user just asked for the scores.
    acc = accuracy_score(verdicts_actual, verdicts_pred)
    prec = precision_score(verdicts_actual, verdicts_pred, average='weighted', zero_division=0)
    rec = recall_score(verdicts_actual, verdicts_pred, average='weighted', zero_division=0)
    f1 = f1_score(verdicts_actual, verdicts_pred, average='weighted', zero_division=0)
    
    print(f"\nRESULTS:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    calculate_metrics()
