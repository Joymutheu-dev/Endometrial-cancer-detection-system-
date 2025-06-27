import os
import argparse
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def visualize_prediction(model Neuropsychiatric, image_path):
    """Visualize model predictions with heatmaps."""
    model = tf.keras.models.load_model(model_path)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) / 255.0
    img_input = np.expand_dims(img, axis=0)
    
    # Predict
    pred = model.predict(img_input)
    class_idx = np.argmax(pred[0])
    class_name = ['Normal', 'Hyperplasia', 'Adenocarcinoma'][class_idx]
    
    # Generate heatmap
    last_conv_layer = model.get_layer('dense')  # Adjust to your model's last dense layer
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0,))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    # Overlay heatmap
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img * 255
    
    # Save visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img.astype(np.uint8))
    plt.title(f"Prediction: {class_name}")
    plt.savefig("prediction_heatmap.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize predictions for endometrial cancer images.")
    parser.add_argument('--model', required=True, help="Path to trained model.")
    parser.add_argument('--image', required=True, help="Path to input image.")
    args = parser.parse_args()
    
    visualize_prediction(args.model, args.image)

if __name__ == "__main__":
    main()