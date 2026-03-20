import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def visualize_weights():
    print("🧠 Extracting Learned Knowledge (Weights/Importance)...")
    
    # Create research directory if it doesn't exist
    if not os.path.exists('research'):
        os.makedirs('research')

    # 1. DL Models (Weight Histograms)
    models_dl = {
        "FNN": "models/fnn/cutoff_fnn_v1.h5",
        "LSTM": "models/lstm/cutoff_lstm_v1.h5",
        "CNN": "models/cnn/cutoff_cnn_v1.keras"
    }

    plt.figure(figsize=(15, 5))
    for i, (name, path) in enumerate(models_dl.items()):
        plt.subplot(1, 3, i+1)
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path, compile=False)
                all_weights = []
                for layer in model.layers:
                    weights = layer.get_weights()
                    if weights:
                        all_weights.extend(weights[0].flatten())
                
                plt.hist(all_weights, bins=50, color='#4f46e5', alpha=0.7)
                plt.title(f"{name}: Weight Distribution", fontsize=12, fontweight='bold')
                plt.yscale('log')
            except Exception as e:
                plt.text(0.5, 0.5, f"Error Loading: {e}", ha='center')
        else:
            plt.text(0.5, 0.5, "Model Not Found", ha='center')
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency (Log)")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('research/dl_weight_histograms.png', dpi=150)
    print("✅ DL Weight Histograms saved to research/dl_weight_histograms.png")

    # 2. Tree Models (Knowledge Mapping / Feature Importance Heatmap)
    try:
        xgb = joblib.load('models/xgboost/cutoff_xgb_v1.joblib')
        rf = joblib.load('models/rf/cutoff_rf_v1.joblib')
        
        # Features (Approximate mapping for visualization)
        features = ['Category', 'Type', 'College', 'State', 'City', 'UniType', 'Course', 'Spec', 'Exam', 'Year', 
                    'EstYear', 'Hostel', 'Academic', 'Faculty', 'Infra', 'Placement', 'HighPkg', 'AvgPkg', 'Fees', 'Duration']
        
        # Ensure feature length matches
        xgb_imp = xgb.feature_importances_[:len(features)]
        rf_imp = rf.feature_importances_[:len(features)]
        
        feat_data = np.vstack([xgb_imp, rf_imp])
        
        plt.figure(figsize=(20, 6), dpi=100)
        sns.heatmap(feat_data, annot=True, fmt=".3f", annot_kws={"size": 9}, cmap='YlGnBu', 
                    xticklabels=features, yticklabels=['XGBoost', 'RandomForest'], cbar_kws={'label': 'Importance Score'})
        
        plt.title("Knowledge Mapping: Feature Sensitivity Heatmap", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Input Features", fontsize=12, labelpad=10)
        plt.tight_layout()
        plt.savefig('research/tree_knowledge_heatmap.png', dpi=150)
        print("✅ Tree Knowledge Heatmap saved to research/tree_knowledge_heatmap.png")
    except Exception as e:
        print(f"⚠️ Error mapping tree knowledge: {e}")

if __name__ == "__main__":
    visualize_weights()
