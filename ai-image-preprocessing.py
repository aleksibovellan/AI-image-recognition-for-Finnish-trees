# AI/ML Trained Image Recognition for Finnish Trees with a Web Interface
# Author: Aleksi Bovellan (2024)


# IMAGE FILE PRE-PROCESSING SCRIPT

# NOTE: This pre-processing script requires a folder 'trees' and its subfolders named after various Finnish trees.
# Place your original tree images in those subfolders.
# Processed images by this script will be saved in a new folder "processed_trees".


# Import necessary libraries
import os
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import cv2
import numpy as np
import subprocess

# Define directories
DATA_DIR = './trees'  # Input folder containing subfolders for each tree species (e.g., 'koivu', 'kuusi')
PROCESSED_DIR = './processed_trees'  # Output folder for processed images

# Main transformation pipeline: resize and apply only essential augmentations
main_transform = transforms.Compose([
    transforms.Resize((224, 224)),         # Standardize size for all images
    transforms.RandomHorizontalFlip(),      # Apply random horizontal flip for augmentation
    transforms.RandomRotation(10),          # Apply slight random rotation
    # ColorJitter removed to maintain original brightness/contrast
])

def pre_preprocess_image(image_path):
    """
    Pre-processes an image to standardize dimensions, color profile, and strip unnecessary metadata.
    This function prepares the image for compatibility with the main transformations.
    Parameters:
        image_path (str): Path to the original image.
    Returns:
        Image: A preprocessed PIL image ready for main transformations, or None if processing failed.
    """
    try:
        # Open the image using Pillow
        with Image.open(image_path) as img:
            # Convert to RGB to ensure compatibility if image isn’t already in RGB
            img = img.convert("RGB") if img.mode != "RGB" else img
            
            # Resize if large to 1024x1024 to reduce memory usage
            if img.width > 1024 or img.height > 1024:
                img.thumbnail((1024, 1024))  # Maintain aspect ratio within 1024x1024 bounds
            
            # Remove EXIF metadata by re-saving the image to a new PIL object
            cleaned_image = Image.new("RGB", img.size)
            cleaned_image.paste(img)

            return cleaned_image  # Return preprocessed image ready for main transformations
    
    except UnidentifiedImageError:
        print(f"Error during pre-preprocessing of {image_path}: cannot identify image file. Attempting OpenCV fallback...")
        
        # Attempt to load and convert the image with OpenCV as a fallback
        return opencv_image_conversion(image_path)

def opencv_image_conversion(image_path):
    """
    Attempts to load and convert problematic images using OpenCV and apply additional pre-processing to make 
    the image compatible with the rest of the script.
    """
    try:
        # Load the image with OpenCV
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            print(f"OpenCV could not load {image_path}. Attempting ImageMagick fallback.")
            return imagemagick_conversion(image_path)
        
        # Convert OpenCV image to PIL format for compatibility with the main transform pipeline
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((1024, 1024))  # Resize if necessary

        return img_pil  # Return the PIL-compatible image for further processing
    
    except Exception as e:
        print(f"OpenCV fallback failed for {image_path}: {e}")
        return imagemagick_conversion(image_path)

def imagemagick_conversion(image_path):
    """
    Uses ImageMagick via subprocess to forcefully convert problematic files into a standard RGB color space.
    This only affects files that failed to load by both PIL and OpenCV.
    """
    try:
        # Apply ImageMagick conversion to RGB color space
        subprocess.run(
            ["mogrify", "-resize", "1024x1024>", "-strip", "-colorspace", "RGB", image_path],
            check=True
        )
        
        # Re-attempt loading the converted image
        with Image.open(image_path) as img:
            img = img.convert("RGB")  # Ensure it is in RGB color space

        print(f"Successfully processed using ImageMagick: {image_path}")
        return img
    
    except Exception as e:
        print(f"ImageMagick fallback failed for {image_path}: {e}")
        return None

def process_images():
    """
    This function processes each image in the tree species subfolders within the 'trees' directory.
    It first applies pre-preprocessing (to clean and standardize each image) before applying main transformations.
    Processed images are saved in the 'processed_trees' folder, maintaining the same folder structure as the input.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        print(f"\nCreated '{PROCESSED_DIR}' directory for processed images.\n")
    
    # Loop through each tree species subfolder
    for species in os.listdir(DATA_DIR):
        species_path = os.path.join(DATA_DIR, species)

        # Skip non-directory files like .DS_Store
        if not os.path.isdir(species_path):
            print(f"Skipping non-directory: {species}")
            continue
        
        # Prepare an output subfolder for the current tree species
        processed_species_path = os.path.join(PROCESSED_DIR, species)
        if not os.path.exists(processed_species_path):
            os.makedirs(processed_species_path)

        print(f"\nProcessing images in '{species}'...\n")

        # Process each image within the current species directory
        for img_name in os.listdir(species_path):
            img_path = os.path.join(species_path, img_name)
            
            # Only process supported image files (jpg, jpeg, png)
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # Pre-preprocess the image to ensure compatibility
                    preprocessed_img = pre_preprocess_image(img_path)
                    if preprocessed_img is not None:
                        # Apply the main transformations
                        img_transformed = main_transform(preprocessed_img)
                        # Save the transformed image to the processed directory
                        save_path = os.path.join(processed_species_path, img_name)
                        img_transformed.save(save_path)
                        print(f"Processed and saved: {img_name}")
                    else:
                        print(f"Skipping incompatible image: {img_name}")
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")
            else:
                print(f"Skipping non-image file: {img_name}")

if __name__ == "__main__":
    print("\n--- Starting Image Preprocessing ---\n")
    process_images()
    print("\n--- Image Preprocessing Completed ---\n")
