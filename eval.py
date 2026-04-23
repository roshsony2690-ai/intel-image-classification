# eval.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("INTEL IMAGE CLASSIFICATION - EVALUATION")
print("="*60)

MODEL_PATH = "models/best_model.h5"
TEST_DIR = "data/seg_test/seg_test"

if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}")
    exit(1)

print("\nLoading model...")
model = load_model(MODEL_PATH)
print("Model loaded!")

print("\nLoading test data...")
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

class_names = list(test_generator.class_indices.keys())
print(f"Found {test_generator.samples} test images")
print(f"Classes: {class_names}")

print("\nEvaluating...")
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"\n{'='*50}")
print(f"TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"TEST LOSS: {loss:.4f}")
print(f"{'='*50}")

print("\nGenerating predictions...")
test_generator.reset()
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

print("\nCLASSIFICATION REPORT:")
print("="*50)
print(classification_report(true_classes, predicted_classes, target_names=class_names))

precision, recall, f1, support = precision_recall_fscore_support(true_classes, predicted_classes, average=None)

print("\nPER-CLASS METRICS:")
print("-"*60)
print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
print("-"*60)
for i, name in enumerate(class_names):
    print(f"{name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<8}")
print("-"*60)
print(f"{'Macro Avg':<15} {np.mean(precision):<12.4f} {np.mean(recall):<12.4f} {np.mean(f1):<12.4f} {np.sum(support):<8}")

print("\nGenerating confusion matrix...")
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={'size': 11})
plt.title('Confusion Matrix - Intel Image Classification', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("Confusion matrix saved as 'confusion_matrix.png'")

print("\nGenerating sample predictions...")
test_generator.reset()
batch_images, batch_labels = next(test_generator)
batch_predictions = model.predict(batch_images, verbose=0)
batch_pred_classes = np.argmax(batch_predictions, axis=1)
batch_true_classes = np.argmax(batch_labels, axis=1)

fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()
correct = 0

for i in range(10):
    axes[i].imshow(batch_images[i])
    true_label = class_names[batch_true_classes[i]]
    pred_label = class_names[batch_pred_classes[i]]
    confidence = np.max(batch_predictions[i]) * 100
    is_correct = true_label == pred_label
    if is_correct:
        correct += 1
    color = 'green' if is_correct else 'red'
    axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%', 
                      color=color, fontsize=9)
    axes[i].axis('off')

plt.suptitle(f'Test Sample Predictions ({correct}/10 correct)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150)
plt.close()
print("Sample predictions saved as 'sample_predictions.png'")

with open('evaluation_results.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("INTEL IMAGE CLASSIFICATION - CNN RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Test Accuracy: {accuracy*100:.2f}%\n")
    f.write(f"Test Loss: {loss:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(true_classes, predicted_classes, target_names=class_names))

print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)
print(f"\nFINAL TEST ACCURACY: {accuracy*100:.2f}%")
print("\nGenerated files:")
print("  • evaluation_results.txt")
print("  • confusion_matrix.png")
print("  • sample_predictions.png")
print("="*60)
