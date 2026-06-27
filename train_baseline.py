import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from models.baseline_cnn import build_baseline_cnn
import os
import glob
import cv2
import numpy as np

def load_data(img_size=(224, 224)):
    real_paths = glob.glob('data/processed_frames/REAL/*.jpg')
    fake_paths = glob.glob('data/processed_frames/FAKE/*.jpg')
    
    X = []
    y = []
    
    for p in real_paths:
        img = cv2.imread(p)
        if img is not None:
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(1) # REAL -> 1
            
    for p in fake_paths:
        img = cv2.imread(p)
        if img is not None:
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(0) # FAKE -> 0
            
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.float32)
    
    # Shuffle the dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    return X[indices], y[indices]

def train():
    X, y = load_data()
    if len(X) == 0:
        print("No data found. Please run generate_dummy_data.py and data_preprocessing.py first.")
        return
        
    print(f"Loaded {len(X)} samples.")
    
    # Simple split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    model = build_baseline_cnn()
    optimizer = Adam(learning_rate=0.0001)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Training Baseline CNN...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=10
    )
    
    os.makedirs('saved_models', exist_ok=True)
    model.save('saved_models/baseline_cnn.h5')
    print("Model saved to saved_models/baseline_cnn.h5")

if __name__ == '__main__':
    train()
