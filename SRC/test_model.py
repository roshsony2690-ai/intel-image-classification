"""
Simple evaluation script for Intel Image Classification CNN model
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents GUI issues
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

print("="*60)
print("INTEL IMAGE CLASSIFICATION - CNN MODEL EVALUATION")
print("="*60)

# Configuration
IMG_SIZE = 150
BATCH_SIZE = 32

# Paths
MODEL_PATH = "models/best_model.h5"
TEST_DIR = "data/seg_test/seg_test"

# 1. Load the model
print("\n[1/5] Loading trained model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# 2. Load test data
print("\n[2/5] Loading test data...")
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
class_names = list(test_generator.class_indices.keys())
print(f"✅ Found {test_generator.samples} test images in {len(class_names)} classes")
print(f"   Classes: {class_names}")

# 3. Evaluate the model
print("\n[3/5] Evaluating model on test set...")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\n{'='*50}")
print(f"🎯 TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"📉 TEST LOSS: {test_loss:.4f}")
print(f"{'='*50}")

# 4. Get detailed predictions
print("\n[4/5] Generating detailed predictions...")
test_generator.reset()
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(true_classes, predicted_classes, average=None)

print("\n📊 PER-CLASS PERFORMANCE:")
print("-" * 60)
print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
print("-" * 60)
for i, class_name in enumerate(class_names):
    print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<8}")
print("-" * 60)
print(f"{'Macro Avg':<15} {np.mean(precision):<12.4f} {np.mean(recall):<12.4f} {np.mean(f1):<12.4f} {np.sum(support):<8}")
print(f"{'Weighted Avg':<15} {np.average(precision, weights=support):<12.4f} {np.average(recall, weights=support):<12.4f} {np.average(f1, weights=support):<12.4f} {np.sum(support):<8}")

# Classification Report
print("\n📋 DETAILED CLASSIFICATION REPORT:")
print("="*60)
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# 5. Save visualizations
print("\n[5/5] Saving visualizations...")

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={'size': 12})
plt.title('Confusion Matrix - Intel Image Classification', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Confusion matrix saved as 'confusion_matrix.png'")

# Sample Predictions (10 random images)
test_generator.reset()
batch_images, batch_labels = next(test_generator)
batch_predictions = model.predict(batch_images, verbose=0)
batch_pred_classes = np.argmax(batch_predictions, axis=1)
batch_true_classes = np.argmax(batch_labels, axis=1)

fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()
correct_count = 0

for i in range(10):
    axes[i].imshow(batch_images[i])
    true_label = class_names[batch_true_classes[i]]
    pred_label = class_names[batch_pred_classes[i]]
    confidence = np.max(batch_predictions[i]) * 100
    is_correct = true_label == pred_label
    if is_correct:
        correct_count += 1
    color = 'green' if is_correct else 'red'
    axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%', 
                      color=color, fontsize=9)
    axes[i].axis('off')

plt.suptitle(f'Test Sample Predictions (Correct: {correct_count}/10)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Sample predictions saved as 'sample_predictions.png'")

# Save all results to a text file
with open('evaluation_results.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("INTEL IMAGE CLASSIFICATION - CNN ASSIGNMENT RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")
    f.write("Per-Class Metrics:\n")
    f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}\n")
    f.write("-"*60 + "\n")
    for i, class_name in enumerate(class_names):
        f.write(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<8}\n")
    f.write("-"*60 + "\n")
    f.write(f"{'Macro Avg':<15} {np.mean(precision):<12.4f} {np.mean(recall):<12.4f} {np.mean(f1):<12.4f} {np.sum(support):<8}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(true_classes, predicted_classes, target_names=class_names))

print("\n✅ Results saved to 'evaluation_results.txt'")

# Final summary
print("\n" + "="*60)
print("✅ EVALUATION COMPLETE!")
print("="*60)
print("\n📁 Generated Files:")
print("   • evaluation_results.txt - Complete text results")
print("   • confusion_matrix.png - Confusion matrix visualization")
print("   • sample_predictions.png - 10 sample predictions")
print(f"\n🏆 FINAL TEST ACCURACY: {test_accuracy*100:.2f}%")
print("="*60)

# Print quick summary
print("\n📊 QUICK SUMMARY:")
print(f"   Best class: {class_names[np.argmax(f1)]} (F1: {np.max(f1):.3f})")
print(f"   Worst class: {class_names[np.argmin(f1)]} (F1: {np.min(f1):.3f})")
print(f"   Overall F1 Score: {np.mean(f1):.3f}")