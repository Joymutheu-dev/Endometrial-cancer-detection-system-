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

2. **Install Python Dependencies**:
   Ensure Python 3.8+ is installed, then install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` includes:
   ```
   tensorflow==2.12.0
   numpy==1.24.3
   opencv-python==4.8.0
   scikit-image==0.20.0
   matplotlib==3.7.2
   openslide-python==1.3.1
   scipy==1.10.1
   pandas==2.0.3
   h5py==3.9.0
   graphviz==0.20.1
   ```

3. **Install Graphviz for Flowchart Visualization**:
   - Install the Graphviz software:
     - **Windows**: Download from https://graphviz.org/download/ and add to PATH.
     - **macOS**: `brew install graphviz`
     - **Linux**: `sudo apt-get install graphviz`
   - This is required for generating the pipeline flowchart (`scripts/visualize_pipeline.py`).

4. **Prepare Dataset**:
   - Place H&E-stained WSI files (e.g., `.svs` format) in the `data/` folder.
   - Organize processed and segmented images in subfolders: `data/processed/` and `data/segmented/`.
   - For classification, create subfolders `data/segmented/normal/`, `data/segmented/hyperplasia/`, and `data/segmented/adenocarcinoma/` with labeled `.png` images.
   - Recommended dataset: TCGA-UCEC (https://portal.gdc.cancer.gov/) or institutional datasets with ethical approval.

5. **Create Output Directories**:
   ```bash
   mkdir -p data/processed data/segmented docs models
   ```

## Usage
The pipeline processes WSIs through preprocessing, segmentation, training, evaluation, and visualization. Below are the instructions to run each component.

### Instructions to Run
1. **Preprocess WSIs**:
   Normalize, denoise, and enhance H&E-stained WSIs:
   ```bash
   python scripts/preprocess.py --input data/ --output data/processed/
   ```
   - **Input**: Directory with `.svs` WSI files.
   - **Output**: Processed `.png` images saved in `data/processed/`.

2. **Segment Images**:
   Apply Otsu thresholding, morphological operations, and watershed algorithm:
   ```bash
   python scripts/segment.py --input data/processed/ --output data/segmented/
   ```
   - **Input**: Directory with processed `.png` images.
   - **Output**: Segmented images saved in `data/segmented/`.

3. **Train the Model**:
   Train the gated MLP model on segmented images:
   ```bash
   python scripts/train.py --data data/segmented/ --model models/saved_model.h5
   ```
   - **Input**: Directory with subfolders for each class (`normal/`, `hyperplasia/`, `adenocarcinoma/`).
   - **Output**: Trained model saved as `models/saved_model.h5`.

4. **Evaluate the Model**:
   Assess model performance on test data:
   ```bash
   python scripts/evaluate.py --model models/saved_model.h5 --data data/segmented/
   ```
   - **Input**: Trained model and test data directory.
   - **Output**: Prints classification report and confusion matrix.

5. **Visualize Predictions**:
   Generate heatmaps for interpretable predictions:
   ```bash
   python scripts/visualize.py --model models/saved_model.h5 --image data/segmented/sample.png
   ```
   - **Input**: Trained model and a segmented image.
   - **Output**: Saves `prediction_heatmap.png` with the original image and heatmap overlay.




