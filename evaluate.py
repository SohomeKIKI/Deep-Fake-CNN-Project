import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from train_baseline import load_data

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'saved_models/{title.replace(" ", "_").lower()}.png')
    plt.close()

def plot_roc_curve(y_true, y_probs, title):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(f'saved_models/{title.replace(" ", "_").lower()}_roc.png')
    plt.close()

def evaluate():
    X, y = load_data()
    if len(X) == 0:
        print("No data found.")
        return
        
    split = int(0.8 * len(X))
    X_val, y_val = X[split:], y[split:]
    
    if os.path.exists('saved_models/baseline_cnn.h5'):
        print("Evaluating Baseline CNN...")
        model = tf.keras.models.load_model('saved_models/baseline_cnn.h5')
        y_probs = model.predict(X_val).flatten()
        y_pred = (y_probs > 0.5).astype(int)
        
        print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
        print(f"Precision: {precision_score(y_val, y_pred):.4f}")
        print(f"Recall: {recall_score(y_val, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")
        print(f"AUC: {roc_auc_score(y_val, y_probs):.4f}")
        
        plot_confusion_matrix(y_val, y_pred, 'Baseline CNN Confusion Matrix')
        plot_roc_curve(y_val, y_probs, 'Baseline CNN')
        
    if os.path.exists('saved_models/cgface.h5'):
        print("\nEvaluating CGFace...")
        model = tf.keras.models.load_model('saved_models/cgface.h5')
        y_probs = model.predict(X_val).flatten()
        y_pred = (y_probs > 0.5).astype(int)
        
        print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
        print(f"Precision: {precision_score(y_val, y_pred):.4f}")
        print(f"Recall: {recall_score(y_val, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")
        print(f"AUC: {roc_auc_score(y_val, y_probs):.4f}")
        
        plot_confusion_matrix(y_val, y_pred, 'CGFace Confusion Matrix')
        plot_roc_curve(y_val, y_probs, 'CGFace')

if __name__ == '__main__':
    evaluate()
