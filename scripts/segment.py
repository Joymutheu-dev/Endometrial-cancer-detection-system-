import os
import argparse
import numpy as np
import cv2
from skimage import morphology, measure, segmentation
from skimage.filters import threshold_otsu

def segment_image(image_path, output_path):
    """Segment WSI using Otsu thresholding and watershed algorithm."""
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Otsu thresholding
    thresh_val = threshold_otsu(gray)
    binary = gray > thresh_val
    
    # Morphological operations
    binary = morphology.remove_small_objects(binary, min_size=100)
    binary = morphology.binary_closing(binary, morphology.disk(3))
    
    # Distance transform and watershed
    dist_transform = cv2.distanceTransform((binary * 255).astype(np.uint8), cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    unknown = cv2.subtract((binary * 255).astype(np.uint8), sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    img_watershed = cv2.watershed(img, markers)
    img[img_watershed == -1] = [255, 0, 0]  # Mark boundaries
    
    # Save segmented image
    output_file = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_file, img)
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Segment WSI images for endometrial cancer analysis.")
    parser.add_argument('--input', required=True, help="Input directory with processed images.")
    parser.add_argument('--output', required=True, help="Output directory for segmented images.")
    args =��.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    for file in os.listdir(args.input):
        if file.endswith('.png'):
            segment_image(os.path.join(args.input, file), args.output)

if __name__ == "__main__":
    main()