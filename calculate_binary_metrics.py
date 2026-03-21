import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import psycopg2
from dotenv import load_dotenv

# Set path
sys.path.append(os.getcwd())
load_dotenv("c:/cutoff-analysis-service/.env")

from app.services.trends.ensemble import CutoffEnsemble

def calculate_binary_metrics():
    print("Calculating Binary Admission Accuracy...")
    ensemble = CutoffEnsemble()
    
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        # Sample more records for statistical significance
        query = "SELECT * FROM ml_features_v3 WHERE closing_rank > 0 ORDER BY RANDOM() LIMIT 200"
        df_test = pd.read_sql(query, conn)
        conn.close()
    except Exception as e:
        print(f"DATABASE FETCH FAILED: {e}")
        return

    y_true = [] # Ground Truth: User Rank <= Actual Rank?
    y_pred = [] # Prediction: User Rank <= Predicted Rank?
    
    for _, row in df_test.iterrows():
        actual = float(row['closing_rank'])
        # Mock historical seq
        history = [actual * 0.95] * 4
        
        try:
            pred = ensemble.get_prediction({'college_name': row['college_name'], 'course_name': row['course_name'], 'exam_name': row['exam_name'], 'category': row['category'], 'year': row['year']}, history_seq=history)
            
            # Simulate a wide-range global user (0.5x to 2.0x of actual cutoff)
            user_rank = actual * np.random.uniform(0.5, 2.0)
            
            y_true.append(1 if user_rank <= actual else 0)
            y_pred.append(1 if user_rank <= pred else 0)
        except:
            continue
            
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract TN, FP, FN, TP and convert to percentages
    total = np.sum(cm)
    tn_pct, fp_pct, fn_pct, tp_pct = (cm.ravel() / total) * 100
    
    print(f"\nRESULTS (GLOBAL PERCENTAGE):")
    print(f"Admission Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nCONFUSION MATRIX (%):")
    print(f"TP: {tp_pct:.1f}%")
    print(f"FP: {fp_pct:.1f}%")
    print(f"TN: {tn_pct:.1f}%")
    print(f"FN: {fn_pct:.1f}%")

if __name__ == "__main__":
    calculate_binary_metrics()
