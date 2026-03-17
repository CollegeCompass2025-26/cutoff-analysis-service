import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import os

def train_cnn():
    print("Loading sequence data for 1D CNN...")
    X = np.load('data/ml_ready/X_lstm.npy')
    y = np.load('data/ml_ready/y_lstm.npy')
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 1D CNN Architecture
    model = Sequential([
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        MaxPooling1D(pool_size=1), # Kernel size is small, max pool 1 to keep dimensions
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("Training CNN...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        verbose=1
    )
    
    os.makedirs('models/cnn', exist_ok=True)
    model.save('models/cnn/cutoff_cnn_v1.keras')
    
    # Save metrics
    with open('research/cnn_metrics.txt', 'w') as f:
        f.write(f"Val MSE: {history.history['val_loss'][-1]}\n")
        f.write(f"Val MAE: {history.history['val_mae'][-1]}\n")
    
    print("CNN Training complete.")

if __name__ == "__main__":
    train_cnn()
