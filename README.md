# Endometrial-cancer-detection-system-

This is a deep learning project for automated endometrial cancer detection and classification from histopathological images using a gated Multi-Layer Perceptron (MLP) model inspired by ECgMLP. Designed for computational biologists, bioinformaticians, and AI-driven cancer researchers, this project leverages H&E-stained whole-slide images (WSIs) to advance endometrial cancer research and clinical applications.

## Overview
This system processes H&E-stained WSIs to detect and classify endometrial cancer (e.g., normal, hyperplasia, adenocarcinoma) with high accuracy, inspired by ECgMLP’s reported 99.26% accuracy. The pipeline includes preprocessing, segmentation, feature extraction, classification, evaluation, and visualization, making it a robust tool for computational biology applications in oncology.

## Features
- **Preprocessing**: Normalization, Non-Local Means (NLM) denoising, and alpha-beta enhancement for WSI preparation.
- **Segmentation**: Otsu thresholding, morphological operations, and watershed algorithm to isolate regions of interest.
- **Classification**: Gated MLP model for multi-class tissue classification (normal, hyperplasia, adenocarcinoma).
- **Evaluation**: Metrics like accuracy, AUROC, and confusion matrix for model performance.
- **Visualization**: Heatmaps for interpretable predictions and a pipeline flowchart for clarity.

### Pipeline Steps
1. **Data Acquisition**: Input H&E-stained WSIs (e.g., TCGA-UCEC).
2. **Preprocessing**: Normalize, denoise (NLM), and enhance (alpha-beta) images.
3. **Segmentation**: Apply Otsu thresholding, morphological operations, and watershed algorithm.
4. **Feature Extraction**: Extract features using a vision backbone (e.g., CNN or ViT).
5. **Classification**: Use gated MLP to classify tissue types.
6. **Evaluation**: Assess performance with accuracy, AUROC, and confusion matrix.
7. **Visualization**: Generate heatmaps to highlight regions contributing to predictions.

## Installation
Follow these steps to set up the EndoCancerVision project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Joymutheu-dev/Endometrialcancerdetectionsystem.git
   cd Endometrialcancerdetectionsystem
