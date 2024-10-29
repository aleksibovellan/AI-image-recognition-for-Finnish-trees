# AI/ML Trained Image Recognition for Finnish trees in Python with Gradio Web Interface

**Human-user interacts through a Gradio Web Interface to upload images of Finnish trees, and the trained AI/ML model recognizes the tree species in the images. Image pre-processing and training requires the user's own provided tree pictures. See further in README for more info.**

**Author:** Aleksi Bovellan (2024)

**Technologies:** ResNet18 pre-trained model from PyTorch, Gradio Web Interface, Python 3.10

**Dataset:** User-provided dataset with images of six tree species: `koivu` (birch), `kuusi` (spruce), `manty` (pine), `pihlaja` (rowan), `tammi` (oak), and `vaahtera` (maple). Data is preprocessed (resized, augmented) using PyTorch's `torchvision.transforms`.

**Python extensions:** torch, torchvision, Pillow, gradio, scikit-learn, numpy, opencv-python

---

![screenshot](https://github.com/user-attachments/assets/4ee9fe24-df6c-4d09-9e61-8649801bc297)

---

# INCLUDED FILES AND FOLDERS:

- "ai-image-preprocessing.py" - This Python script pre-processes your own original image files of Finnish trees, so that they will work with the training algorithm. It expects a folder "trees" in the same directory, which has subfolders for each Finnish tree name. Then add your own pictures of trees into their relevant subfolders. Those original image files should be in JPG, JPEG, or PNG formats, and preferably in RGB color space or something similarly simple as that - otherwise the pre-processing script might skip some files if they are in a too large and complicated color space. When finished, the pre-processing script finally creates a new folder "processed_trees", which will house all the new processed image files.

- "ai-image-training.py" - This is the Python script for the automated training process after image pre-processing is already done. The training results are stored in a new file "tree_species_model.pth". 

- "ai-image-interface.py" - Python script for the model and Gradio web interface - it serves a local web page.

- Folder "trees" - This folder should include subfolders for various Finnish tree names, and your original tree images in them. The image pre-processing script needs this folder structure and files to build from. The original images are not provided in this repository, but the folder structure is just in case.

- "README.md" - this file for instructions


# SPECIAL ATTENTION TO APPLE MAC COMPUTERS:

For easier usage, you could use the Macs Terminal window instead of any code editors. First create a virtual environment in the Terminal by typing:

python3 -m venv myenv

source myenv/bin/activate

pip install torch torchvision Pillow gradio scikit-learn numpy opencv-python


(When you are done experimenting with this project, you can exit the virtual environment by typing "deactivate", and later return into it by repeating "source myenv/bin/activate")


# PRE-INSTALLATION REQUIREMENTS FOR ALL USERS:

pip install torch torchvision Pillow gradio scikit-learn numpy opencv-python

**ImageMagick:** Ensure ImageMagick is installed for additional image processing compatibility. This tool is used in this script to handle any images that might not normally get loaded to be processed with the standard libraries due to the image's unsupported and probably larger color-spaces or sizes.

- Mac: Install with brew install imagemagick
- Linux: Install with sudo apt-get install imagemagick
- Windows: Download and install from ImageMagick’s official site


# USAGE:

1. Ensure that your directory has all the provided 3 x Python scripts, and a folder "trees", with subfolders named after Finnish trees, and your own original tree images in those subfolders.

2. Run the image pre-processing script: python3 ai-image-preprocessing.py

3. Run the automated training script: python3 ai-image-training.py

4. Start the Gradio web interface: python3 ai-image-interface.py

5. Use the locally served Gradio web page interface to upload an example picture of a tree and press button "SUBMIT" for the result.


# Editing the training script - epoch value to adjust the training time:

Example: Setting an epoch value from "20" to "100" takes around 10 minutes on a year 2020 MacBook M1 Air.


# The training script and evaluation results:

**Validation Accuracy:**

Accuracy measures how many of the model’s predictions are correct. It's one of the most direct metrics for classification tasks.

**Expected ranges:**

0-40% accuracy: Indicates the model might not be learning effectively or is overfitting (learning the training data but not generalizing).

40-60% accuracy: Indicates moderate performance but likely with some class imbalances or overfitting issues.

60-80% accuracy: A good range indicating the model is making solid predictions but still has room for improvement.

80%+ accuracy: Strong performance, indicating the model is well-tuned and likely generalizing well.

**F1-Score:**

The F1-score balances precision and recall, making it a useful metric when you have class imbalances.

**Expected ranges:**

0-0.3: Weak performance—model may be struggling with class imbalance or not generalizing well.

0.3-0.6: Moderate performance—model is learning, but certain classes might be harder to predict.

0.6-1.0: Strong performance—model is predicting well across most classes.


# CPU/GPU COMPATIBILITY

The training script runs using whatever the host computer automatically decides to use (CPU/GPU) for best performance.


---

# Pre-process Script Flowchart

![Solutions-ImagePreprocess-flowchart](https://github.com/user-attachments/assets/b6784952-34ae-4b2e-9f7a-fa544fc1a603)

---

# Training Script Flowchart

![Solutions-Training-flowchart](https://github.com/user-attachments/assets/faace207-8ea0-4bc7-9046-200c5ad1d863)

---

# Gradio Interface Script Flowchart

![Solutions-Interface-flowchart](https://github.com/user-attachments/assets/9e5a8115-cf69-481e-805c-6c556da1b2fa)

---
