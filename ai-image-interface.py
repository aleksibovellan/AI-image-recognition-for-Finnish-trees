# AI/ML Trained Image Recognition for Finnish trees with a Web Interface
# Author: Aleksi Bovellan (2024)


# AI/ML MODEL AND MAIN GRADIO WEB INTERFACE SCRIPT

# Requires a folder 'processed_trees' resulted from running the image pre-processing script beforehand.
# Requires training results in the file "tree_species_model.pth"


# Import necessary libraries
import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
from torchvision.models import ResNet18_Weights

# Load the trained model for 6 tree species
model = models.resnet18(weights=None)  # Load the architecture without pretrained weights
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 6)  # Ensure 6 output classes (koivu, kuusi, manty, pihlaja, tammi, vaahtera)
model.load_state_dict(torch.load('tree_species_model.pth', weights_only=True))  # Secure loading
model.eval()  # Set model to evaluation mode

# Use the same device-agnostic method as training (auto-detect GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define image transformation (resize and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Tree species class names (should match the subfolder names in your dataset)
class_names = ['koivu', 'kuusi', 'manty', 'pihlaja', 'tammi', 'vaahtera']

# Prediction function for Gradio
def predict(image):
    """
    This function takes an input image, transforms it, and predicts the tree species.
    """
    image = Image.fromarray(image.astype('uint8'), 'RGB')  # Convert NumPy array to PIL image
    image = transform(image).unsqueeze(0).to(device)  # Apply transformations and move to device
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        return class_names[pred]

# Gradio interface setup
interface = gr.Interface(fn=predict, inputs="image", outputs="text", title="AI/ML Trained Image Recognition for Finnish trees with a Web Interface")

# Main execution
if __name__ == "__main__":
    print("\n--- Launching Gradio Interface ---\n")
    interface.launch()  # Launch the Gradio web interface
    print("\nGradio Interface closed.\n")
