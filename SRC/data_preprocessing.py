import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def explore_dataset(data_dir):
    """Explore dataset structure"""
    train_dir = os.path.join(data_dir, 'seg_train/seg_train')
    test_dir = os.path.join(data_dir, 'seg_test/seg_test')
    
    class_names = sorted(os.listdir(train_dir))
    print("="*50)
    print("DATASET EXPLORATION")
    print("="*50)
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    # Count images per class
    print("\nImages per class (training):")
    for class_name in class_names:
        class_path = os.path.join(train_dir, class_name)
        num_images = len(os.listdir(class_path))
        print(f"  {class_name}: {num_images} images")
    
    # Check sample image
    sample_class_path = os.path.join(train_dir, class_names[0])
    sample_image_path = os.path.join(sample_class_path, os.listdir(sample_class_path)[0])
    sample_img = cv2.imread(sample_image_path)
    sample_img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    
    print(f"\nSample image size: {sample_img.shape}")
    
    # Display sample image
    plt.figure(figsize=(6, 6))
    plt.imshow(sample_img_rgb)
    plt.title(f"Sample {class_names[0]} image")
    plt.axis('off')
    plt.show()
    
    return class_names, train_dir, test_dir

def create_data_generators(train_dir, img_height=150, img_width=150, batch_size=32):
    """Create data generators with augmentation"""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    
    return train_generator, validation_generator

def create_test_generator(test_dir, img_height=150, img_width=150, batch_size=32):
    """Create test data generator"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator

if __name__ == "__main__":
    # Test the preprocessing
    data_dir = "data"
    class_names, train_dir, test_dir = explore_dataset(data_dir)
    train_gen, val_gen = create_data_generators(train_dir)
    test_gen = create_test_generator(test_dir)
    print("\nData generators created successfully!")