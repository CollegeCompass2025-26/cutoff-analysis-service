import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os

def train_lstm():
    print("Loading sequence data...")
    X = np.load('data/ml_ready/X_lstm.npy')
    y = np.load('data/ml_ready/y_lstm.npy')
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Dataset summary: {len(X_train)} train, {len(X_test)} test")
    
    # Simple LSTM Architecture
    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("Training LSTM...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        verbose=1
    )
    
    os.makedirs('models/lstm', exist_ok=True)
    model.save('models/lstm/cutoff_lstm_v1.h5')
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('LSTM Training Progress')
    plt.legend()
    plt.savefig('research/lstm_training_loss.png')
    
    print("LSTM Training complete.")

if __name__ == "__main__":
    train_lstm()
