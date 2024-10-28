# AI/ML Trained Image Recognition for Finnish trees with a Web Interface
# Author: Aleksi Bovellan


# IMAGE FILE PRE-PROCESSING SCRIPT

# NOTE: The pre-processing of the tree images for the learning has ALREADY been done - they are in the folder "processed_trees". So there's no need to pre-process anything.
# However, if pre-processing is needed for new images, this pre-processing script requires a folder 'trees' and its sub folders named after various Finnish trees, and the images in those sub folders.
# Original tree images are not included in this repository's "trees" folder, but the empty folder structure for it is provided just in case.


# Import necessary libraries
import os
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import cv2
import numpy as np

# Define directories
DATA_DIR = './trees'  # Input folder containing subfolders for each tree species (e.g., 'koivu', 'kuusi')
PROCESSED_DIR = './processed_trees'  # Output folder for processed images

# Transformation pipeline: resize, augment, and convert to tensor
main_transform = transforms.Compose([
    transforms.Resize((224, 224)),         # Standardize size for all images
    transforms.RandomHorizontalFlip(),      # Apply random horizontal flip
    transforms.RandomRotation(10),          # Apply random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness and contrast
])

def pre_preprocess_image(image_path):
    """
    Pre-processes an image to standardize dimensions, color profile, and strip unnecessary metadata.
    This function aims to prepare the image for compatibility with the main transformations.
    Parameters:
        image_path (str): Path to the original image.
    Returns:
        Image: A preprocessed PIL image ready for main transformations, or None if processing failed.
    """
    try:
        # Open the image using Pillow
        with Image.open(image_path) as img:
            # Convert to RGB to ensure compatibility
            img = img.convert("RGB")
            
            # Resize large images to a maximum of 1024x1024 to save memory and ensure consistency
            if img.width > 1024 or img.height > 1024:
                img.thumbnail((1024, 1024))  # Maintain aspect ratio within 1024x1024 bounds
            
            # Remove EXIF data by re-saving the image to a new PIL object
            cleaned_image = Image.new("RGB", img.size)
            cleaned_image.paste(img)

            return cleaned_image  # Return preprocessed image ready for main transformations
    
    except UnidentifiedImageError:
        print(f"Error during pre-preprocessing of {image_path}: cannot identify image file. Attempting OpenCV fallback...")
        
        # Attempt to load the image with OpenCV as a fallback
        try:
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                print(f"OpenCV could not load {image_path}. Skipping file.")
                return None
            
            # Convert OpenCV image to PIL format for compatibility with the main transform pipeline
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((1024, 1024))  # Resize if necessary

            return img_pil  # Return the PIL-compatible image for further processing
        
        except Exception as e:
            print(f"Fallback failed for {image_path}: {e}")
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