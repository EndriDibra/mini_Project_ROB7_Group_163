# Author: Endri Dibra 
# Project: ARP mini Project 

# Importing the required libraries  
import re
import cv2
import torch 
import heapq
import random  
import numpy as np
from PIL import Image
from ultralytics import YOLO
from scipy.interpolate import splprep, splev  
from transformers import AutoProcessor, AutoModelForVision2Seq


# Defining path of the occupancy grid map image
mapImage = "Test_Images/Laboratory.jpg" 

# Defining output path for saving final image
outputPath = "New_Results/ImageResultLaboratory.png"

# Defining output path for saving final social cost map image
outputPath2 = "New_Results/Social_Cost_Map_Laboratory.png"

# System requirements setup
# Loading the VLM model
# particularly smolVLM-Instruct 
SOCIAL_MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"

# Initializing the model processor (tokenizer and feature extractor)
socialProcessor = AutoProcessor.from_pretrained(SOCIAL_MODEL_ID)

# Initializing the VLM model and moving it to CPU memory
socialModel = AutoModelForVision2Seq.from_pretrained(SOCIAL_MODEL_ID).to("cpu")

# Setting the model to evaluation mode
socialModel.eval()

# Values for obstacle severity penalty
PERSON_MULTI = 1.0
OBSTACLE_MULTI  = 0.3


# This function is used to create a
# social cost map with smolVLM contribution
def Social_Cost_Map(img, output_size):
    
    # Converting input image to PIL format for model input
    pilIMG = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Constructing messages for chat template with image and prompt
    messages = [
    
        {
            "role": "user",
            "content": [
                
                {"type": "image"},
               
                {"type": "text", "text": "Top-down room. Score social cost 0-1 for 4 quadrants: top-left, top-right, bottom-left, bottom-right. 0.0=safe, 1.0=bad near yellow people and obstacles. Output ONLY floating point numbers, no extra text."}

            ]
        }
    ]

    # Applying chat template to generate prompt string
    prompt = socialProcessor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Processing text and image inputs for model inference
    inputs = socialProcessor(text=prompt, images=pilIMG, return_tensors="pt").to("cpu")

    # Generating output with no gradient computation for efficiency
    with torch.no_grad():
    
        # Generating token IDs using model with specified parameters
        output_ids = socialModel.generate(
    
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.3,
        )

    # Decoding generated token IDs to text string
    text = socialProcessor.decode(output_ids[0], skip_special_tokens=True)
    
    # Isolating Assistant
    # Converting text to lowercase for finding assistant section
    textLower = text.lower()
    
    # Checking and extracting assistant response if present
    if "assistant:" in textLower:
    
        # Finding start position of assistant response
        assistant_start = textLower.find("assistant:") + len("assistant:")
    
        # Extracting raw output from assistant section
        raw_VLM_Output = text[assistant_start:].strip()
    
    # Fallback extraction if no assistant marker found
    else:
    
        # Removing prompt and stripping whitespace
        raw_VLM_Output = text.replace(prompt, "").strip()
    
    # Cleaning VLM output by removing special tokens
    clean_VLM_Output = re.sub(r'<[^>]+>', '', raw_VLM_Output).strip()
    
    # Initializing scores variable
    scores = None
    
    # Searching for bracketed quadrant values using regex
    match = re.search(r'\[([^\]]+)\]', clean_VLM_Output)  
    
    # Processing matched group if found
    if match:
    
        # Attempting to parse quadrant string
        try:
    
            # Removing spaces from matched string
            quadSTR = match.group(1).replace(' ', '')
    
            # Splitting and converting to floats, filtering non-empty
            quads = [float(x.strip()) for x in quadSTR.split(',') if x.strip()]
    
            # Using first 4 if available
            if len(quads) >= 4:
    
                scores = quads[:4]
    
            # Padding with 0.5 if fewer than 4
            elif len(quads) > 0:
    
                scores = quads + [0.5] * (4 - len(quads))
    
        # Handling parsing errors silently
        except ValueError:
    
            pass
    
    # Enhanced Fallback: Any floats/ints from output
    # Applying fallback if no scores parsed yet
    if not scores:
    
        # Finding all numeric values in cleaned output
        allNums = re.findall(r'\d+\.?\d*', clean_VLM_Output)
    
        # Processing found numbers if any
        if allNums:
    
            # Attempting to convert to floats
            try:
    
                # Converting first 4 numbers to floats
                quads = [float(n) for n in allNums[:4]]
    
                # Clamping scores to [0,1] range and padding if needed
                scores = [min(max(q, 0.0), 1.0) for q in quads] + [0.5] * (4 - len(quads))  
    
            # Handling conversion errors
            except ValueError:
    
                pass
    
    # Defaulting to neutral scores if still no valid output
    if not scores:
    
        # Printing warning for default usage
        print("No valid quadrants from VLM, defaulting to 0.5")
    
        # Setting default quadrant scores
        scores = [0.5, 0.5, 0.5, 0.5]
    
        # Updating raw output for logging
        raw_VLM_Output = str(scores)
    
    # Printing parsed quadrant costs for debugging
    print(f"Quadrant costs: {scores}")
    
    # Creating 2x2 array from quadrant scores
    quadArray = np.array([[scores[0], scores[1]], [scores[2], scores[3]]], dtype=np.float32)
    
    # Resizing array to output dimensions using cubic interpolation
    arrayResized = cv2.resize(quadArray, output_size, interpolation=cv2.INTER_CUBIC)
    
    # Normalizing resized array if max value exceeds 0
    if arrayResized.max() > 0:
    
        # Dividing by max to scale to [0,1]
        arrayResized /= arrayResized.max()

    # Returning normalized array and raw output string
    return arrayResized, raw_VLM_Output


# Defining function for evaluating
# path social awareness and optimality
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
    inputs = socialProcessor(text=prompt, images=pilIMG, return_tensors="pt").to("cpu")

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


# This function is for VLM queries to provide a score for each detected obstacle
def VLM_Object_Threat(img, boxes): 
    
    # Converting input image to PIL format for cropping
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Initializing empty list for threat scores
    scores = []

    # Iterating over each bounding box
    for (x1, y1, x2, y2) in boxes:
    
        # Cropping image to bounding box region
        crop = pil.crop((x1, y1, x2, y2))

        # Constructing messages for chat template with cropped image and prompt
        messages = [
    
            {
                "role": "user",
                "content": [
    
                    {"type": "image"},
    
                    {
                        "type": "text",
                        "text": "Rate SOCIAL-RISK of this object on scale 0–1. 0=safe, 1=highest risk. Output ONLY a floating point number, no extra text"
                        "."
                    }
                ]
            }
        ]

        # Applying chat template to generate prompt string
        prompt = socialProcessor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Processing text and cropped image inputs for model inference
        inputs = socialProcessor(text=prompt, images=crop, return_tensors="pt").to("cpu")

        # Generating output with no gradient computation for efficiency
        with torch.no_grad():
    
            # Generating token IDs using model with specified parameters
            ids = socialModel.generate(
    
                **inputs, max_new_tokens=10, do_sample=True, temperature=0.3
            )

        # Decoding generated token IDs to text
        text = socialProcessor.decode(ids[0], skip_special_tokens=True)

        # Extract number
        # Finding all numeric values in decoded text
        nums = re.findall(r"\d+\.?\d*", text)
    
        # Processing first number if found
        if nums:
    
            # Converting to float and clamping to [0,1]
            v = float(nums[0])
            v = min(max(v, 0.0), 1.0)
    
        # Fallback to neutral score if no number found
        else:
    
            v = 0.5   
    
        # Appending computed score to list
        scores.append(v)

    # Returning list of threat scores
    return scores


# Reference coordinates (defined in 500x500 space) 
# Reference width for scaling coordinates
refWidth = 500

# Reference height for scaling coordinates
refHeight = 500

# Defining the starting point coordinate in the 500x500 reference space
refStart = [400, 480]   

# Defining the goal point coordinate in the 500x500 reference space
refGoal  = [345, 235]     

# Dynamic: Reading actual image to get real size 
# Temporarily reading the map image to check its actual dimensions
tempImage = cv2.imread(mapImage)

# Checking if image loaded successfully
if tempImage is None:

    # Raising an error if the map file is not found
    raise FileNotFoundError(f"Image not found: {mapImage}")

# Getting the actual height and width of the loaded image
actualHeight, actualWidth = tempImage.shape[:2]

# Defining a fixed test size for the planner resolution
testWidth, testHeight = 250, 250

# Applying test size for the planning map dimensions
plannerWidth = testWidth
plannerHeight = testHeight

# Printing forced test dimensions
print(f"Forced test size: {plannerWidth}×{plannerHeight}")

# Scale start/goal to actual image size 
# Calculating the scaling factor for the X dimension
scaleX = plannerWidth / refWidth

# Calculating the scaling factor for the Y dimension
scaleY = plannerHeight / refHeight

# Scaling the reference start X/Y coordinates to the planner size
startPixel = [int(refStart[0] * scaleX), int(refStart[1] * scaleY)]

# Scaling the reference goal X/Y coordinates to the planner size
goalPixel  = [int(refGoal[0]  * scaleX), int(refGoal[1]  * scaleY)]

# Printing image and scaled point information
print(f"Image loaded: {actualWidth}x{actualHeight}")
print(f"Scaled Start: {startPixel}, Goal: {goalPixel}")

# Scaling visual elements
# Scaling the marker radius based on the planner size
markerRadius = max(1, int(7 * scaleX))        

# Scaling the path line thickness
lineThickness = max(1, int(2 * scaleX))       

# Scaling the bounding box outline thickness
boxThickness = max(1, int(2 * scaleX))        

# Defining path drawing color (BGR: Blue, Green, Red)
pathColor = (0, 0, 255)

# Defining start marker color (BGR: Green)
startColor = (0, 255, 0)

# Defining goal marker color (BGR: Blue)
goalColor = (255, 0, 0)

# Setting maximum cost distance for obstacles
maxCostDistance = 20

# Setting cost amplifier multiplier for obstacle proximity costs
costAmplifier = 5

# Setting safety radius around obstacles, scaled to planner size
safetyRadius = max(1, int(9  * scaleX))   # at least 1

# Setting B-spline smoothing factor
# scaled to planner size, scaling down
smoothingFactor = max(2, int(20 * min(scaleX, scaleY)))  

# Setting penalty for turning changes, used in A*
turnPenalty = 0.5

# Setting weight of curvature penalty, used in A*
curvatureWeight = 0.5

# Setting margin distance from walls, scaled to planner size
wallMargin = max(1, int(11 * scaleX))

# Defining YOLO model path for object detection
yoloModelPath = "yolo11n.pt"


# Defining function for computing Euclidean distance
def Heuristic(a, b):

    # Calculating squared differences in X and Y
    dx = (a[0] - b[0])**2
    dy = (a[1] - b[1])**2 
    
    # Returning the Euclidean distance (straight line distance)
    return np.sqrt(dx + dy)


# Defining function for shifting a point inside wall margins
def ShiftInsideWall(point):

    # Unpacking point coordinates
    x, y = point

    # Clamping X within wall margins (min(max(x, min_bound), max_bound))
    x = min(max(x, wallMargin), plannerWidth - wallMargin - 1)

    # Clamping Y within wall margins
    y = min(max(y, wallMargin), plannerHeight - wallMargin - 1)

    # Returning shifted point
    return [x, y]


# Defining function for reconstructing path from cameFrom dictionary
def ReconstructPath(cameFrom, start, goal):

    # Initializing path with goal point
    path = [goal]

    # Setting current point to goal
    current = goal

    # Tracing back until reaching start using the parent dictionary
    while current != start:

        # Moving to parent node
        current = cameFrom[current]

        # Appending current node to path
        path.append(current)

    # Reversing path to get start-to-goal order
    return path[::-1]


# Defining A* search function with turn and curvature penalties
def AStar(grid, costMap, start, goal):

    # Initializing priority queue with (f_cost, g_cost, current_node, parent, grandparent)
    priorityQueue = [(0, 0, start, None, None)]

    # Initializing dictionary to store parent node for path reconstruction
    cameFrom = {start: start}

    # Initializing dictionary to store g-score (cost from start)
    gScore = {start: 0}

    # Getting grid shape
    rows, cols = grid.shape

    # Defining movement directions (dx, dy, baseCost) for 8 directions
    moves = [

        (0, 1, 1), (0, -1, 1), (1, 0, 1), (-1, 0, 1), # Cardinal moves (cost 1)
        (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)),      # Diagonal moves (cost sqrt(2))
        (-1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2))
    ]

    # Looping while queue is not empty
    while priorityQueue:

        # Popping node with lowest f cost: f = g + h
        f, g, current, prev, prevPrev = heapq.heappop(priorityQueue)

        # Checking if goal is reached
        if current == goal:

            # Returning reconstructed path
            return ReconstructPath(cameFrom, start, goal)

        # Unpacking current coordinates
        x, y = current

        # Iterating over possible moves
        for dx, dy, baseCost in moves:

            # Calculating next position
            nx = x + dx
            ny = y + dy

            # Checking bounds and free cell (grid[y, x] == 1 means free space)
            if 0 <= nx < cols and 0 <= ny < rows: 

                # Validating grid cell is traversable
                if grid[ny, nx] == 1:

                    # Initializing turn penalty
                    turnCost = 0

                    # Calculating turn penalty: checks if the move direction changed
                    if prev is not None:

                        # Compares (current - prev) vector to (next - current) vector
                        if (current[0] - prev[0], current[1] - prev[1]) != (nx - x, ny - y):

                            # Applying turn penalty if direction changes
                            turnCost = turnPenalty

                    # Initializing curvature penalty
                    curvatureCost = 0

                    # Calculating curvature penalty if applicable (needs 3 points: prevPrev, prev, current)
                    if prev is not None and prevPrev is not None:

                        # Computing vector from prevPrev to prev
                        v1 = np.array([prev[0] - prevPrev[0], prev[1] - prevPrev[1]])

                        # Computing vector from current to next
                        v2 = np.array([nx - x, ny - y])

                        # Normalizing vectors to unit length for stable cosine
                        if np.linalg.norm(v1) > 0:
                           
                            # Normalizing first vector
                            v1 = v1 / np.linalg.norm(v1)
                        
                        # Normalizing second vector if non-zero
                        if np.linalg.norm(v2) > 0:
                        
                            v2 = v2 / np.linalg.norm(v2)

                        # Calculating angle if vectors are non-zero
                        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:

                            # Computing the angle of the two vectors 
                            cosAngle = np.dot(v1, v2)

                            # Clipping cosine to valid range [-1, 1] to prevent math errors
                            cosAngle = np.clip(cosAngle, -1, 1)

                            # Calculating angle (turn magnitude)
                            angle = np.arccos(cosAngle)

                            # Applying curvature weight to the angle
                            curvatureCost = curvatureWeight * angle

                    # Calculating total g score: distance + map cost + turn penalty + curvature penalty
                    newG = g + baseCost + costMap[ny, nx] + turnCost + curvatureCost

                    # Defining neighbor node
                    neighbor = (nx, ny)

                    # Updating path if better g score found (standard A* update check)
                    if neighbor not in gScore or newG < gScore[neighbor]:

                        # Storing new g score
                        gScore[neighbor] = newG

                        # Storing parent node
                        cameFrom[neighbor] = current

                        # Pushing neighbor to priority queue with its f cost, g cost, current, and previous parent
                        heapq.heappush(priorityQueue, (newG + Heuristic(neighbor, goal), newG, neighbor, current, prev))

    # Returning None if no path found
    return None


# Defining function for checking line of sight between two points
def LineOfSight(grid, p1, p2):

    # Unpacking first point (x1, y1)
    x1, y1 = p1

    # Unpacking second point (x2, y2)
    x2, y2 = p2

    # Calculating absolute differences in coordinates
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    # Initializing current position
    x = x1
    y = y1
    
    # Calculating total steps based on Bresenham's algorithm
    n = 1 + dx + dy
    
    # Calculating step increments (direction of travel)
    xInc = 1 if x2 > x1 else -1
    yInc = 1 if y2 > y1 else -1
    
    # Initializing error term for Bresenham's line algorithm
    error = dx - dy
    
    # Doubling differences for error calculation
    dx *= 2
    dy *= 2
    
    # Iterating over line points
    for k in range(n):
        
        # Returning False if hitting obstacle (grid cell is 0)
        if grid[y, x] == 0:
            
            return False
        
        # Updating position in X or Y based on error term
        if error > 0:
            
            # Incrementing X and subtracting Y error
            x += xInc
            error -= dy
            
        # Incrementing Y and adding X error
        else:
            
            y += yInc
            error += dx
    
    # Returning True if line is clear (no obstacles encountered)
    return True


# Defining function for pruning unnecessary path points
def PrunePath(path, grid):

    # Returning empty list if path is empty
    if len(path) == 0:

        return []

    # Initializing pruned path list with the starting point
    pruned = [path[0]]

    # Initializing index of the current segment start point
    index = 0

    # Looping until the current index reaches the last point
    while index < len(path) - 1:

        # Setting nextIndex to the end of the remaining path segment
        nextIndex = len(path) - 1

        # Searching backward from the end for the furthest point with line of sight
        while nextIndex > index + 1:

            # Checking for direct line of sight between the current point and nextIndex
            if LineOfSight(grid, path[index], path[nextIndex]):

                break

            # Moving one step backward if line of sight fails
            nextIndex -= 1

        # Appending next visible point (which is the new segment end point)
        pruned.append(path[nextIndex])

        # Updating index to the new segment end point
        index = nextIndex

    # Returning pruned path
    return pruned


# Defining function for smoothing path using B-spline
def SmoothPath(path, smoothFactor=5):

    # Returning original path if too short for spline calculation
    if len(path) < 4:

        return path

    # Converting path list of tuples to NumPy array
    pts = np.array(path)

    # Extracting X coordinates
    xs = pts[:, 0]

    # Extracting Y coordinates
    ys = pts[:, 1]

    # Creating parametric B-spline representation (tck: knots, coefficients, degree)
    tck, u = splprep([xs, ys], s=smoothFactor)

    # Creating dense parameter array for smooth evaluation
    uNew = np.linspace(0, 1, len(path))

    # Evaluating spline at dense parameter points
    out = splev(uNew, tck)

    # Converting to integer points and zipping X and Y coordinates
    smoothed = list(zip(out[0].astype(int), out[1].astype(int)))

    # Clamping to bounds and shifting inside walls
    smoothed = [(min(max(int(x), 0), plannerWidth-1), min(max(int(y), 0), plannerHeight-1)) for x, y in smoothed]
    smoothed = [ShiftInsideWall(p) for p in smoothed]

    # Returning smoothed path
    return smoothed


# Defining main function for running planner
def RunPlanner():

    # Reading map image
    image = cv2.imread(mapImage)

    # Checking if image is loaded
    if image is None:

        # Printing error message for missing map
        print("Map not found")

        return

    # Resizing image to planner dimensions
    image = cv2.resize(image, (plannerWidth, plannerHeight))

    # Copying image for drawing the path and markers
    imgDraw = image.copy()

    # Loading YOLO model for object detection
    model = YOLO(yoloModelPath)

    # Detecting obstacles
    # Running YOLO on fixed 320x320 for stable detection
    yoloSize = 320
    
    # Calculating aspect-preserving scale
    h, w = image.shape[:2]

    # Computing resize scale to fit within YOLO input size
    scale = min(yoloSize / w, yoloSize / h)
    
    # Calculating new dimensions after scaling
    newW, newH = int(w * scale), int(h * scale)
    
    # Resizing image to new dimensions
    yoloImage = cv2.resize(image, (newW, newH))
    
    # Calculating horizontal padding
    padW = (yoloSize - newW) // 2
    
    # Calculating vertical padding
    padH = (yoloSize - newH) // 2
    
    # Padding image to square YOLO input size
    yoloImage = cv2.copyMakeBorder(yoloImage, padH, yoloSize - newH - padH, padW, yoloSize - newW - padW, cv2.BORDER_CONSTANT, 0)
    
    # Running YOLO detection on padded image
    results = model(yoloImage, save=False)

    # Initializing list of bounding boxes
    boxes = []
        
    # Setting a minimum size for detected boxes, scaled
    minBoxSize = max(8, int(25 * scaleX))  # Stronger min size

    # Processing YOLO detection results
    for r in results:
            
        # Iterating over all detected bounding boxes
        for box, conf, cls in zip(
                                    r.boxes.xyxy.cpu().numpy(),
                                    r.boxes.conf.cpu().numpy(),
                                    r.boxes.cls.cpu().numpy()
                                ):

            # Skipping low-confidence detections
            if conf < 0.35:
            
                continue
                            
            # Unpacking coordinates (x_min, y_min, x_max, y_max)
            x1, y1, x2, y2 = box.astype(float)

            # Correcting padding offsets
            x1 -= padW
            y1 -= padH
            x2 -= padW
            y2 -= padH
            
            # Rescaling X coordinates to planner size
            x1 = x1 * plannerWidth  / newW
            x2 = x2 * plannerWidth  / newW
            
            # Rescaling Y coordinates to planner size
            y1 = y1 * plannerHeight / newH
            y2 = y2 * plannerHeight / newH

            # Converting coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Calculating width and height
            bw, bh = x2 - x1, y2 - y1

            # Enforcing minimum size on width
            if bw < minBoxSize:
                    
                # Calculating center for width expansion
                center = (x1 + x2) // 2
                
                # Expanding box to minimum width around center
                x1 = center - minBoxSize // 2
                x2 = center + (minBoxSize + 1) // 2
                
            # Enforcing minimum size on height
            if bh < minBoxSize:
                    
                # Calculating center for height expansion
                center = (y1 + y2) // 2
                
                # Expanding box to minimum height around center
                y1 = center - minBoxSize // 2
                y2 = center + (minBoxSize + 1) // 2

            # Ensuring minimum 1-pixel size
            if x2 <= x1: x2 = x1 + 1
            
            if y2 <= y1: y2 = y1 + 1

            # Clipping coordinates to be within image bounds
            x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, [plannerWidth-1, plannerHeight-1]*2)
            
            # Appending final box coordinates
            boxes.append(((int(x1), int(y1), int(x2), int(y2)), int(cls)))

    # Filling grid
    # Initializing obstacle grid (0: free, 1: obstacle)
    grid = np.zeros((plannerHeight, plannerWidth), dtype=np.uint8)
        
    # Filling the grid with detected bounding boxes (set to 1)
    for (x1, y1, x2, y2), cid in boxes:
            
        # Marking bounding box region as obstacle
        grid[y1:y2, x1:x2] = 1

    # Applying small dilation to slightly expand detected obstacles
    extraKernel = np.ones((3, 3), np.uint8)
    grid = cv2.dilate(grid, extraKernel, iterations=1)

    # Calculating dilation scale based on safety radius and planner size
    dilateScale = max(1.5, 2.0 * scaleX)
        
    # Determining kernel size for safety radius dilation (must be odd)
    kernelSize = max(3, int(safetyRadius * dilateScale * 2 + 1))
    kernelSize = kernelSize if kernelSize % 2 == 1 else kernelSize + 1
        
    # Creating the dilation kernel
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
        
    # Applying strong dilation to create inflated obstacles
    # Proportional inflation
    iterations = max(1, int(safetyRadius / (kernelSize // 2))) 
    inflated = cv2.dilate(grid, kernel, iterations=iterations)

    # Displaying the grid map with white masks,
    # that show people are puffier than other obstacles,
    # thus, increased safety around them 
    cv2.imshow("Inflated Grid Map", (inflated * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Defining free cells: planner grid (1: free, 0: obstacle)
    planner = (inflated == 0).astype(np.uint8)

    # Applying wall margins (setting border cells to obstacle/0)
    planner[:wallMargin, :] = 0
    planner[-wallMargin:, :] = 0
    planner[:, :wallMargin] = 0
    planner[:, -wallMargin:] = 0

    # Shifting start and goal points to ensure they are inside wall margins
    start = ShiftInsideWall(startPixel)
    goal = ShiftInsideWall(goalPixel)

    # Protecting start and goal from dilation
    # Defining protection radius around start/goal
    protectionRadius = max(3, int(15 * scaleX))

    # Creating coordinate grids
    yGrid, xGrid = np.ogrid[:plannerHeight, :plannerWidth]

    # Protecting start point
    startX, startY = start
        
    # Creating a circular mask around the start point
    startMask = (xGrid - startX)**2 + (yGrid - startY)**2 <= protectionRadius**2
        
    # Setting masked areas in planner grid to free (1)
    planner[startMask] = 1
        
    # Setting masked areas in inflated grid to non-obstacle (0)
    inflated[startMask] = 0

    # Protecting goal point
    goalX, goalY = goal
        
    # Creating a circular mask around the goal point
    goalMask = (xGrid - goalX)**2 + (yGrid - goalY)**2 <= protectionRadius**2
        
    # Setting masked areas in planner grid to free (1)
    planner[goalMask] = 1
        
    # Setting masked areas in inflated grid to non-obstacle (0)
    inflated[goalMask] = 0

    # Creating binary obstacle map (1: obstacle, 0: free)
    binaryObs = (inflated != 0).astype(np.uint8)

    # Calculating distance transform (distance to nearest obstacle)
    distance = cv2.distanceTransform(1 - binaryObs, cv2.DIST_L2, 5)

    # Clipping distance values to a maximum for cost calculation stability
    distance = np.clip(distance, 0.1, maxCostDistance)

    # Creating cost map
    # Stronger cost near obstacles
    # Applying inverse-distance cost function (high cost near obstacles, low far away)
    costMap = costAmplifier * (1.0 / (distance + 1.0))**3
        
    # Capping cost map for stability
    costMap = np.clip(costMap, 0, 100) 

    # Checking on cost map if people are shown as larger
    # white blobs versus oher obstacles
    cv2.imshow("Cost Map", (costMap / costMap.max() * 255).astype(np.uint8))  
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Querying VLM for social cost
    print("Querying VLM for social cost...")
        
    # Getting social cost map from VLM and its raw output
    socialCost, vlmReasoning = Social_Cost_Map(image, (plannerWidth, plannerHeight))

    # Hierarchical: YOLO local threats on VLM quadrants
    # Initializing threat map as zeros
    threatMap = np.zeros((plannerHeight, plannerWidth), dtype=np.float32)

    # Querying VLM for per-object threat
    print("Querying VLM for per-object threat...")
    objScores = VLM_Object_Threat(image, [b for (b, _cid) in boxes])

    print("The scores area:", objScores)

    # Processing each box and score for threat map
    for ((x1, y1, x2, y2), cid), score in zip(boxes, objScores):

        # Scaling score by class
        # person class for YOLO11
        if cid in [0]:   
          
            # Applying person multiplier to score
            score *= PERSON_MULTI
        
        # Applying chair multiplier to score
        else:
        
            score *= OBSTACLE_MULTI

        # Calculating center coordinates
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Creating coordinate grids for distance calculation
        yy, xx = np.ogrid[:plannerHeight, :plannerWidth]

        # Computing Euclidean distance grid from center
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)

        # Setting sigma for Gaussian decay
        sigma = safetyRadius

        # Computing Gaussian decay weighted by score
        decay = np.exp(-dist / sigma) * score

        # Taking maximum across overlapping threats
        threatMap = np.maximum(threatMap, decay)  

    # Clipping threat map to [0,1] range
    threatMap = np.clip(threatMap, 0, 1)

    # Setting alpha for blending social costs
    # Blending: 60% VLM global, 40% YOLO local
    alpha = 0.4 

    # Blending global and local social costs
    socialCost = (1 - alpha) * socialCost + alpha * threatMap

    # Printing maximum threat value
    print(f"Hierarchical blend: Max threat {threatMap.max():.2f}")

    # Setting weight for the social cost component
    socialWeight = 3.5
        
    # Adding the weighted social cost to the overall cost map
    costMap += socialWeight * socialCost

    # Adding wall extra cost
    for i in range(wallMargin):

        # Calculating cost based on distance to the wall margin
        wallCost = (wallMargin - i) / wallMargin * costAmplifier * 10

        # Applying wall cost to all four sides
        costMap[i, :] += wallCost
        costMap[-i - 1, :] += wallCost
        costMap[:, i] += wallCost
        costMap[:, -i - 1] += wallCost


    # Defining function for ensuring valid points
    def EnsureValid(points):

        # Unpacking point coordinates
        x, y = points

        # Initializing attempt counter
        attempts = 0

        # Looping until free cell is found or attempts exceed 100
        while planner[y, x] == 0 and attempts < 100:
                
            # Jittering every 10 steps
            if attempts % 10 == 0:
            
                # Adding random jitter to coordinates
                x += random.randint(-1, 1)
            
                y += random.randint(-1, 1)

            # Shifting X towards the nearest image half
            if x < plannerWidth / 2:

                # Incrementing X if left half
                x += 1

            # Decrementing X if right half
            else:

                x -= 1

            # Shifting Y towards the nearest image half
            if y < plannerHeight / 2:

                # Incrementing Y if top half
                y += 1

            # Decrementing Y if bottom half
            else:

                y -= 1

            # Ensuring the shifted point is still inside wall margins
            x, y = ShiftInsideWall([x, y])

            # Incrementing attempt counter
            attempts += 1

        # Returning the validated free cell point
        return (x, y)


    # Validating start point (s) to ensure it's in a free cell
    start = EnsureValid(tuple(start))

    # Validating goal point (g) to ensure it's in a free cell
    goal = EnsureValid(tuple(goal))

    # Printing start and goal coordinates
    print("Start", start, "Goal", goal)

    # Running A* search
    print("Running A*")
        
    # Executing A* to find the initial raw path
    raw = AStar(planner, costMap, start, goal)

    # Checking if path exists
    if raw is None:
    
        # Printing message for no path found
        print("No path found, trying straight line")
        
        # Checking direct line of sight for fallback path
        if LineOfSight(planner, start, goal):
        
            # Creating direct path from start to goal
            path = [start, goal]

            # Using direct path as raw for post-processing
            raw = path  
            
        # Handling case with no viable path
        else:
        
            # Printing no viable path message
            print("No viable path")
            return

    # Pruning path to remove redundant waypoints
    raw = PrunePath(raw, planner)

    # Smoothing path using B-spline interpolation
    path = SmoothPath(raw, smoothingFactor)

    # Shifting final path points inside walls (last safety check)
    path = [ShiftInsideWall(point) for point in path]

    # Drawing path lines
    for i in range(len(path) - 1):

        # Drawing line segment between adjacent path points
        cv2.line(imgDraw, tuple(path[i]), tuple(path[i + 1]), pathColor, lineThickness)
        
    # Drawing start circle marker
    cv2.circle(imgDraw, start, markerRadius, startColor, -1)

    # Drawing goal circle marker
    cv2.circle(imgDraw, goal, markerRadius, goalColor, -1)

    # Drawing detected boxes
    for (x1, y1, x2, y2), cid in boxes:
        
        # Drawing the yellow bounding box outline
        cv2.rectangle(imgDraw, (x1, y1), (x2, y2), (0, 255, 255), boxThickness)

    # Saving output image
    cv2.imwrite(outputPath, imgDraw)

    # Printing saved message
    print("Saved", outputPath)

    # VLM Reasoning and Evaluation 
    # Printing VLM social cost reasoning header
    print("\nVLM SOCIAL COST REASONING")
    
    # Printing raw VLM reasoning output
    print(vlmReasoning)

    # Printing VLM path evaluation header
    print("\nVLM PATH EVALUATION")
        
    # Executing VLM path evaluation function
    evaluationResult = EvaluatePath(imgDraw, start, goal)
    
    # Printing evaluation result
    print(evaluationResult)

    # Visualizing the social cost map
    # Scaling social cost to 8-bit grayscale for display
    visualization = (socialCost * 255).astype(np.uint8)
    
    # Displaying social cost map window
    cv2.imshow("Social Cost", visualization)

    # Saving social cost map image
    cv2.imwrite(outputPath2, visualization)

    # Printing saved message for social cost map
    print("Saved", outputPath2)

    # Showing A* planner window 
    cv2.imshow("A* Planner", imgDraw)

    # Waiting for key press to close windows
    cv2.waitKey(0)

    # Closing all windows
    cv2.destroyAllWindows()


# Running main logic
if __name__ == "__main__":
    
    # Executing planner main function
    RunPlanner()
