import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import os
from tensorflow.keras.models import load_model

def evaluate_model(model, test_generator, class_names):
    """Evaluate model on test set"""
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Reset generator
    test_generator.reset()
    
    # Get predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Calculate accuracy
    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    # Per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(true_classes, predicted_classes, average=None)
    
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    print("\nPer-class Metrics:")
    print(metrics_df.to_string(index=False))
    
    return true_classes, predicted_classes, accuracy

def plot_confusion_matrix(true_classes, predicted_classes, class_names):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 10})
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nConfusion matrix saved to 'plots/confusion_matrix.png'")

def visualize_predictions(test_generator, model, class_names, num_samples=10):
    """Visualize predictions on random test images"""
    
    # Get a batch of test images
    test_generator.reset()
    batch_images, batch_labels = next(test_generator)
    
    # Get predictions
    predictions = model.predict(batch_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(batch_labels, axis=1)
    
    # Plot random samples
    plt.figure(figsize=(15, 8))
    
    indices = np.random.choice(len(batch_images), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)
        plt.imshow(batch_images[idx])
        
        true_label = class_names[true_classes[idx]]
        pred_label = class_names[predicted_classes[idx]]
        confidence = np.max(predictions[idx]) * 100
        
        color = 'green' if true_label == pred_label else 'red'
        
        plt.title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%', 
                  color=color, fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('plots/predictions_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nPredictions visualization saved to 'plots/predictions_visualization.png'")

if __name__ == "__main__":
    print("Evaluation module loaded")