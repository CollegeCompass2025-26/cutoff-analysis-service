import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import joblib

def prepare_rnn_data():
    print("Preparing data for RNN (Temporal sequence)...")
    df = pd.read_csv('data/ml_ready/train_features.csv') # Simplified for now
    y = pd.read_csv('data/ml_ready/train_target.csv').values
    
    # For a true RNN, we should sequence by College+Course over Years
    # For now, let's treat it as a Deep Neural Network with Embeddings 
    # to demonstrate the DL pipeline, then refine for sequences.
    
    return train_test_split(df, y, test_size=0.2, random_state=42)

def build_fnn_model(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1) # Predict log rank
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_dl():
    X_train, X_test, y_train, y_test = prepare_rnn_data()
    
    print("Building Deep Learning Model...")
    model = build_fnn_model(X_train.shape[1])
    
    print("Training FNN model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=128,
        verbose=1
    )
    
    os.makedirs('models/fnn', exist_ok=True)
    model.save('models/fnn/cutoff_fnn_v1.h5')
    
    # Save training history for research paper
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('research/fnn_training_history.csv', index=False)
    
    print("FNN Training complete.")

if __name__ == "__main__":
    train_dl()
