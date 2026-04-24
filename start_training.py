import sys
import os

# Add SRC to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SRC'))

# Import all required modules
from data_preprocessing import explore_dataset, create_data_generators
from model import create_cnn_model
from train import train_model

# Set data path - FIXED for your structure
data_dir = "data"

print("=" * 50)
print("INTEL IMAGE CLASSIFICATION - CNN TRAINING")
print("=" * 50)

# 1. Load and explore dataset
print("\n[1/4] Loading dataset...")
class_names, train_dir, test_dir = explore_dataset(data_dir)
print(f"✓ Found {len(class_names)} classes: {class_names}")

# 2. Create data generators
print("\n[2/4] Creating data generators...")
train_gen, val_gen = create_data_generators(train_dir)
test_gen = create_test_generator(test_dir)
print("✓ Data generators ready")

# 3. Build and train model
print("\n[3/4] Building and training CNN model...")
model = create_cnn_model(len(class_names))
history = train_model(model, train_gen, val_gen)

# 4. Evaluate
print("\n[4/4] Evaluating model...")
test_loss, test_acc = model.evaluate(test_gen)
print(f"\n✓ Test Accuracy: {test_acc:.4f}")

print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("=" * 50)
