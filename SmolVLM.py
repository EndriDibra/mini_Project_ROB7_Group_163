# Author: Endri Dibra 
# Project: ARP mini Project 

# Importing the required libraries
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


# System requirements setup
# Loading the VLM model
# particularly smolVLM-Instruct 
MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"

# The Occupancy grid map
# image to be used for the process
IMAGE_PATH = "Test_Images/Occupancy_Grid_Map.png"

# Using either CPU or GPU of the
# computer based on availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Maximun number of tokens to be
# used from smolVLM for responses
MAX_NEW_TOKENS = 200

# New size of the occupancy grid map image
# for computing resourse management
NEW_SIZE = (384, 384)  

# Loading smolVLM model 
print(f"Loading model {MODEL_ID} on device: {DEVICE}...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

print("Model loaded successfully.")

# Checking if image exists
if os.path.exists(IMAGE_PATH):

    # Loading image and converting image from RGB to BGR
    image = Image.open(IMAGE_PATH).convert("RGB") 

    # Resizing image
    image = image.resize(NEW_SIZE)

    print(f"Loaded image: {IMAGE_PATH}")

# Case where image does not exists
else:

    print(f"Error! Image not found.")

    exit()

# Preparing prompt for smolVLM
prompt = "Describe in one detailed paragraph the objects detected in this image." \
          "Elaborate your responses"

# Enabling the right prompt format 
# feeded to smolVLM processing process 
formattedPrompt = f"<image>{prompt}" 

# Processing an input pair of image and prompt
inputs = processor(images=image, text=formattedPrompt, return_tensors="pt").to(DEVICE)

print("Generating response...")

# Creating an output, a response 
with torch.no_grad():

    outputIDS = model.generate(

        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False, 
        temperature=0.0
    )

# Decocing output text results 
outputText = processor.decode(outputIDS[0], skip_special_tokens=True)
description = outputText.replace(formattedPrompt, "").strip()

# Displaying output text
print("\nModel Output")
print(description) 