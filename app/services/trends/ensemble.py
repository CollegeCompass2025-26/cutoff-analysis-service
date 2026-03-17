import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os

class CutoffEnsemble:
    def __init__(self):
        # Load Models
        print("Loading ensemble models...")
        self.xgb = joblib.load('models/xgboost/cutoff_xgb_v1.joblib')
        self.rf = joblib.load('models/rf/cutoff_rf_v1.joblib')
        
        # Using compile=False for inference to avoid custom object/metric errors
        self.lstm = tf.keras.models.load_model('models/lstm/cutoff_lstm_v1.keras', compile=False)
        self.fnn = tf.keras.models.load_model('models/fnn/cutoff_fnn_v1.h5', compile=False)
        self.cnn = tf.keras.models.load_model('models/cnn/cutoff_cnn_v1.keras', compile=False)
        
        self.encoders = joblib.load('models/encoders/label_encoders.joblib')
        
    def get_prediction(self, features, history_seq=None):
        """
        Features: dict with college, course, category, year
        history_seq: list of last 3 years cutoffs [2021, 2022, 2023]
        """
        # 1. Feature Prep
        encoded_feats = []
        for col in ['exam_name', 'round', 'college_name', 'course_name', 'category', 'quota', 'gender', 'institute_type']:
            val = features.get(col, 'Unknown')
            try:
                enc_val = self.encoders[col].transform([val])[0]
            except:
                enc_val = 0 # Default/Fallback
            encoded_feats.append(enc_val)
        encoded_feats.append(features.get('year', 2024))
        
        # 2. Static Prediction (XGBoost)
        xgb_pred = self.xgb.predict([encoded_feats])[0]
        
        # 3. Temporal Prediction (LSTM)
        if history_seq is not None:
            seq = np.log1p(np.array(history_seq).reshape(1, 3, 1))
            lstm_pred = self.lstm.predict(seq)[0][0]
            # Fusion: Weight temporal models higher if history exists
            final_log_rank = 0.7 * lstm_pred + 0.3 * xgb_pred
        else:
            final_log_rank = xgb_pred
            
        final_rank = np.expm1(final_log_rank)
        return int(final_rank)

    def get_risk_assessment(self, final_rank, user_rank):
        """Feature 9: Seat Risk Assessment"""
        delta = user_rank - final_rank
        if delta < -500: return "SAFE"
        if delta < 500: return "MODERATE"
        return "RISKY"

if __name__ == "__main__":
    # Quick Test
    ensemble = CutoffEnsemble()
    test_feats = {
        'college_name': 'IIT Bombay',
        'course_name': 'Computer Science',
        'category': 'General',
        'year': 2024
    }
    pred = ensemble.get_prediction(test_feats, history_seq=[200, 210, 195])
    print(f"Ensemble Predicted Rank: {pred}")
    print(f"Risk for rank 250: {ensemble.get_risk_assessment(pred, 250)}")
