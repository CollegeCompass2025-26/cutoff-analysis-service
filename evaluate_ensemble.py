import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
import psycopg2
from dotenv import load_dotenv
import traceback

# Set path
sys.path.append(os.getcwd())
load_dotenv("c:/cutoff-analysis-service/.env")

from app.services.trends.ensemble import CutoffEnsemble

def evaluate_ensemble_comprehensively():
    print("Initializing Ensemble for Exhaustive Evaluation...")
    try:
        ensemble = CutoffEnsemble()
    except Exception as e:
        print(f"FAILED TO INITIALIZE ENSEMBLE: {e}")
        return
    
    # 1. Fetch a representative test sample from the DB (ml_features_v3)
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        # Get diverse categories and years
        query = """
        SELECT * FROM ml_features_v3 
        WHERE closing_rank > 0 
        ORDER BY RANDOM() 
        LIMIT 500
        """
        df_test = pd.read_sql(query, conn)
        conn.close()
        print(f"Sampled {len(df_test)} records from DB.")
    except Exception as e:
        print(f"DATABASE FETCH FAILED: {e}")
        return
        
    actuals = []
    predictions = []
    verdicts_actual = [] 
    verdicts_pred = []
    
    success_count = 0
    error_log = []

    for idx, row in df_test.iterrows():
        # Prepare features for get_prediction
        # We MUST use the names expected by ensemble.py
        feat = {
            'college_name': row['college_name'],
            'course_name': row['course_name'],
            'specialization_name': row.get('specialization', 'Unknown') or 'Unknown',
            'exam_name': row['exam_name'],
            'category': row['category'],
            'cutoff_type': 'rank',
            'state': row.get('state', 'Unknown') or 'Unknown',
            'city': row.get('city', 'Unknown') or 'Unknown',
            'typeofuni': row.get('typeofuni', 'Unknown') or 'Unknown',
            'year': row['year'],
            'established_year': row.get('established_year', 0) or 0,
            'rating_academic': row.get('rating_academic', 0) or 0,
            'rating_placement': row.get('rating_placement', 0) or 0,
            'avg_package': row.get('avg_package', 0) or 0,
            'fees': row.get('fees', 0) or 0
        }
        
        # History seq (simulate last 4 rounds)
        actual_rank = float(row['closing_rank'])
        history = [actual_rank * 0.85, actual_rank * 0.9, actual_rank * 0.95, actual_rank * 0.98]
        
        try:
            pred = ensemble.get_prediction(feat, history_seq=history)
            
            # Simulated User Rank for Verdicts
            user_rank = actual_rank * 1.05
            
            p_v = ensemble.get_risk_assessment(pred, user_rank)
            a_v = ensemble.get_risk_assessment(actual_rank, user_rank)
            
            actuals.append(actual_rank)
            predictions.append(float(pred))
            verdicts_pred.append(p_v)
            verdicts_actual.append(a_v)
            success_count += 1
            
        except Exception as e:
            if len(error_log) < 3:
                error_log.append(f"Row {idx} Failed: {str(e)}")
            continue

    print(f"Total Successful Predictions: {success_count}")
    if error_log:
        print("Sample Errors:")
        for err in error_log: print(f"  {err}")

    if success_count < 10:
        print("ERROR: Too few successful predictions to generate metrics. Check for categorical encoding errors.")
        return

    # --- Metrics Calculation ---
    y_true = np.log1p(np.array(actuals))
    y_pred = np.log1p(np.array(predictions))
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"\n--- FINAL ENSEMBLE METRICS ---")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE (Log): {mae:.4f}")

    # --- Visualizations ---
    os.makedirs('research/ensemble_eval', exist_ok=True)
    
    # Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.3, color='#3498db')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', lw=2)
    plt.xscale('log'); plt.yscale('log')
    plt.title(f'Ensemble Accuracy (R²={r2:.3f})')
    plt.savefig('research/ensemble_eval/accuracy_scatter.png')

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(verdicts_actual, verdicts_pred, labels=["SAFE", "MODERATE", "RISKY"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=["SAFE", "MODERATE", "RISKY"], yticklabels=["SAFE", "MODERATE", "RISKY"])
    plt.title('Admission Verdict Confusion Matrix')
    plt.ylabel('Actual Verdict'); plt.xlabel('Predicted Verdict')
    plt.savefig('research/ensemble_eval/confusion_matrix.png')

    print(f"Dashboard saved to research/ensemble_eval/")

if __name__ == "__main__":
    evaluate_ensemble_comprehensively()
