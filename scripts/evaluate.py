import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model_path, data_dir, target_size=(224, 224)):
    """Evaluate the trained model on test data."""
    model = tf.keras.models.load_model(model_path)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate
    predictions = model.predict(test_generator)
    y_true = test_generator.classes
    y_pred = np.argmax(predictions, axis=1)
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def main():
    parser = argparse.ArgumentParser(description="Evaluate gated MLP for endometrial cancer classification.")
    parser.add_argument('--model', required=True, help="Path to trained model.")
    parser.add_argument('--data', required=True, help="Directory with test images.")
    args = parser.parse_args()
    
    evaluate_model(args.model, args.data)

if __name__ == "__main__":
    main()