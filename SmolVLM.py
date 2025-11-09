# Author: Endri Dibra 
# Project: ARP mini Project 

# Importing the required libraries
import os
import re
import cv2
import torch 
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


# System requirements setup
# Loading the VLM model
# particularly smolVLM-Instruct 
SOCIAL_MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"

# Defining path of the occupancy grid map image
mapImage = "New_Results/ImageResult.png"

# Initializing the model processor (tokenizer and feature extractor)
socialProcessor = AutoProcessor.from_pretrained(SOCIAL_MODEL_ID)

# Initializing the VLM model and moving it to device memory
device = "cuda" if torch.cuda.is_available() else "cpu"

socialModel = AutoModelForVision2Seq.from_pretrained(SOCIAL_MODEL_ID).to(device)

# Setting the model to evaluation mode
socialModel.eval()

# Printing model loading information
print(f"Loading model {SOCIAL_MODEL_ID} on device: {device}...")
print("Model loaded successfully.")

# Checking if image exists
if os.path.exists(mapImage):

    # Loading image using OpenCV for flexibility
    img = cv2.imread(mapImage)
    
    if img is None:
        
        raise ValueError(f"Could not load image: {mapImage}")
    
    # Converting image from BGR to RGB color channels
    pilIMG = Image.open(mapImage).convert("RGB") 

    # Printing loaded image confirmation
    print(f"Loaded image: {mapImage}")

# Raising an error if the map file is not found
else:

    raise FileNotFoundError(f"Image not found: {mapImage}")


# This function evaluates the path in the image using VLM
def EvaluatePath(img, start, goal): 
    
    # Converting input image to PIL format for model input
    pilIMG = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Constructing messages for chat template with image and prompt
    messages = [
    
        {
            "role": "user",
            "content": [
              
                {"type": "image"},
                
                {"type": "text", "text": "You are given an image with a red path, a green start point, a blue goal point, and yellow obstacles. Score from 1 to 5 range: socially aware navigation and speed of the red path. Output ONLY integer numbers in brackets [], no extra text."}

            ]
        }
    ]

    # Applying chat template to generate prompt string
    prompt = socialProcessor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Processing text and image inputs for model inference
    inputs = socialProcessor(text=prompt, images=pilIMG, return_tensors="pt").to(device)

    # Generating output with no gradient computation for efficiency
    with torch.no_grad():
    
        # Generating token IDs using model with specified parameters
        outputIDs = socialModel.generate(
    
            **inputs,
            max_new_tokens=15,
            do_sample=True,
            temperature=0.3,
        )

    # Decoding generated token IDs to evaluation text
    evaluationText = socialProcessor.decode(outputIDs[0], skip_special_tokens=True)
    
    # Isolate Assistant
    # Converting text to lowercase for finding assistant section
    textLower = evaluationText.lower()
    
    # Checking and extracting assistant response if present
    if "assistant:" in textLower:
    
        # Finding start position of assistant response
        assistant_start = textLower.find("assistant:") + len("assistant:")
    
        # Extracting raw output from assistant section
        raw_VLM_Output = evaluationText[assistant_start:].strip()
    
    # Fallback extraction if no assistant marker found
    else:
    
        # Removing prompt and stripping whitespace
        raw_VLM_Output = evaluationText.replace(prompt, "").strip()
    
    # Cleaning VLM output by removing special tokens
    clean_VLM_Output = re.sub(r'<[^>]+>', '', raw_VLM_Output).strip()
    
    # Finding all integer values after cleaning
    allInts = re.findall(r'\d+', clean_VLM_Output.replace('.', ''))  
    
    # Processing found integers if at least one exists
    if len(allInts) >= 1:  
    
        # Clamping scores to [1,5] range using first 2
        scores = [max(1, min(5, int(i))) for i in allInts[:2]]
    
        # Padding with average score if fewer than 2
        if len(scores) < 2:
    
            scores += [3] * (2 - len(scores))  
    
    # Defaulting to neutral scores if no integers found
    else:
    
        scores = [3, 3]

    # Returning formatted evaluation string if scores available
    if scores:
    
        # Formatting scores into multi-line string
        return f"""
               1. Social Awareness: {scores[0]} / 5
               2. Optimality/Speed: {scores[1]} / 5
               """
    
    # Handling fallback case for evaluation
    else:
    
        # Printing fallback message
        print("Eval fallback to average")
    
        # Setting default scores
        scores = [3, 3]
    
        # Returning formatted fallback string
        return f"""
               1. Social Awareness: {scores[0]} / 5 (fallback)
               2. Optimality/Speed: {scores[1]} / 5 (fallback)
               """


# Querying VLM for path evaluation
print("Querying VLM for path evaluation...")

# Defining the starting point coordinate in the 250x250 image space 
start = [242, 125]

# Defining the goal point coordinate in the 250x250 image space
goal = [7, 220]

# Getting evaluation from VLM
evaluation = EvaluatePath(img, start, goal)

# Printing model output
print("\nModel Output")
print(evaluation)
