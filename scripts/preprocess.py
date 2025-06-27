import os
import argparse
import numpy as np
import cv2
from skimage import io, filters
import openslide

def preprocess_image(image_path, output_path, patch_size=224):
    """Preprocess H&E-stained WSI: normalization, NLM denoising, alpha-beta enhancement."""
    # Load WSI
    slide = openslide.OpenSlide(image_path)
    img = slide.read_region((0, 0), 0, (patch_size, patch_size)).convert('RGB')
    img = np.array(img)
    
    # Normalization
    img = img / 255.0
    
    # Non-Local Means denoising
    img_denoised = cv2.fastNlMeansDenoisingColored((img * 255).astype(np.uint8))
    img_denoised = img_denoised / 255.0
    
    # Alpha-beta enhancement
    alpha, beta = 1.2, 0.1
    img_enhanced = np.clip(alpha * img_denoised + beta, 0, 1)
    
    # Save processed image
    output_file = os.path.join(output_path, os.path.basename(image_path).replace('.svs', '.png'))
    io.imsave(output_file, (img_enhanced * 255).astype(np.uint8))
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Preprocess WSI images for endometrial cancer analysis.")
    parser.add_argument('--input', required=True, help="Input directory with WSI images.")
    parser.add_argument('--output', required=True, help="Output directory for processed images.")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    for file in os.listdir(args.input):
        if file.endswith('.svs'):
            preprocess_image(os.path.join(args.input, file), args.output)

if __name__ == "__main__":
    main()