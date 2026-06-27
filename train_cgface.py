import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from models.cgface import build_cgface
from train_baseline import load_data
import os

def train():
    X, y = load_data()
    if len(X) == 0:
        print("No data found.")
        return
        
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    model = build_cgface()
    optimizer = Adam(learning_rate=0.00005)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("Training CGFace Model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=16, # As per report
        epochs=25,
        callbacks=[lr_reduction, early_stopping]
    )
    
    os.makedirs('saved_models', exist_ok=True)
    model.save('saved_models/cgface.h5')
    print("Model saved to saved_models/cgface.h5")

if __name__ == '__main__':
    train()
