import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.gated_mlp import build_gated_mlp

def load_data(data_dir, target_size=(224, 224)):
    """Load and preprocess images for training."""
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical'
    )
    return generator

def main():
    parser = argparse.ArgumentParser(description="Train gated MLP for endometrial cancer classification.")
    parser.add_argument('--data', required=True, help="Directory with segmented images (subfolders: normal, hyperplasia, adenocarcinoma).")
    parser.add_argument('--model', required=True, help="Path to save trained model.")
    args = parser.parse_args()
    
    # Load data
    train_generator = load_data(args.data)
    
    # Build and train model
    model = build_gated_mlp(input_shape=(224, 224, 3), num_classes=train_generator.num_classes)
    model.fit(train_generator, epochs=20, verbose=1)
    
    # Save model
    model.save(args.model)

if __name__ == "__main__":
    main()