import os
os.environ['KERAS_BACKEND'] = 'tensorflow' # Explicitly set for Keras 3 compatibility
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

class CutoffEnsemble:
    def __init__(self):
        # Load Models
        print("Loading ensemble models...")
        self.xgb = joblib.load('models/xgboost/cutoff_xgb_v1.joblib')
        self.rf = joblib.load('models/rf/cutoff_rf_v1.joblib')
        
        # Using compile=False for inference to avoid custom object/metric errors
        try:
            self.lstm = tf.keras.models.load_model('models/lstm/cutoff_lstm_v1.h5', compile=False)
            self.fnn = tf.keras.models.load_model('models/fnn/cutoff_fnn_v1.h5', compile=False)
            self.cnn = tf.keras.models.load_model('models/cnn/cutoff_cnn_v1.keras', compile=False)
        except Exception as e:
            print(f"Warning: Deep learning models failed to load ({e}). Using fallback logic.")
            self.lstm = self.fnn = self.cnn = None
        
        self.encoders = joblib.load('models/encoders/label_encoders.joblib')
        
    def get_prediction(self, features, history_seq=None):
        """
        Features: dict with data matching the ml_features_v2 schema
        history_seq: list of last 4 cutoff rounds [R1, R2, R3, R4]
        """
        # 1. Feature Prep (Must match DataPreprocessor.prepare_features Exactly)
        categorical_cols = [
            'category', 'cutoff_type', 'college_name', 'state', 'city', 
            'typeofuni', 'course_name', 'specialization_name', 'exam_name'
        ]
        numeric_features = [
            'established_year', 'rating_hostel', 'rating_academic', 'rating_faculty', 
            'rating_infra', 'rating_placement', 'highest_package', 'avg_package', 
            'fees', 'duration_years'
        ]
        
        input_vector = []
        
        # Add encoded categoricals
        for col in categorical_cols:
            val = features.get(col, 'Unknown')
            try:
                enc_val = self.encoders[col].transform([str(val)])[0]
            except:
                enc_val = 0
            input_vector.append(enc_val)
            
        # Add numerics (default to 0)
        for col in numeric_features:
            input_vector.append(float(features.get(col, 0)))
            
        # Add year
        input_vector.append(float(features.get('year', 2025)))
        
        # ENSURE EXACTLY 20 FEATURES FOR XGBOOST (Match Training Schema)
        # 9 (Cat) + 10 (Num) + 1 (Year) = 20
        input_vector = input_vector[:20]
        
        # 2. Static Predictions (XGBoost & Random Forest)
        try:
            xgb_pred = self.xgb.predict([input_vector])[0]
            rf_pred = self.rf.predict([input_vector])[0]
        except Exception as e:
            xgb_pred = 7.0 # Fallback log rank
            rf_pred = 7.0
        
        # 3. Temporal Prediction (LSTM)
        if history_seq is not None and len(history_seq) >= 4:
            # Take exactly the last 4 rounds for our sequence
            seq_data = history_seq[-4:]
            seq = np.log1p(np.array(seq_data).reshape(1, 4, 1))
            lstm_pred = self.lstm.predict(seq, verbose=0)[0][0]
            
            # 4. Hybrid Fusion (All 3 Models)
            # Weights: 50% XGBoost, 20% Random Forest, 30% LSTM
            final_log_rank = 0.5 * xgb_pred + 0.2 * rf_pred + 0.3 * lstm_pred
        else:
            # Fallback to Tree-based average if no history
            final_log_rank = 0.7 * xgb_pred + 0.3 * rf_pred
            
        final_rank = np.expm1(final_log_rank)
        return int(final_rank)

    def get_risk_assessment(self, final_rank, user_rank):
        """
        Feature 9: Seat Risk Assessment using a 10% Relative Confidence Interval.
        A fixed-rank delta (e.g. 500) fails at extreme ends (Rank 50 vs Rank 50,000).
        A 10% relative delta provides a consistent strategic "cushion" across all categories.
        """
        threshold = 0.10 * final_rank
        delta = user_rank - final_rank
        
        if delta < -threshold:
            return "SAFE"
        elif delta < threshold:
            return "MODERATE"
        else:
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
    pred = ensemble.get_prediction(test_feats, history_seq=[200, 210, 195, 190])
    print(f"Ensemble Predicted Rank: {pred}")
    print(f"Risk for rank 250: {ensemble.get_risk_assessment(pred, 250)}")
