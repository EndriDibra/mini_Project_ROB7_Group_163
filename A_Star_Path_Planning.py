# Author: Endri Dibra 
# Project: ARP mini Project 

# Importing the required libraries  
import re
import cv2
import json
import torch 
import heapq
import numpy as np
from PIL import Image
from ultralytics import YOLO
from scipy.interpolate import splprep, splev  
from transformers import AutoProcessor, AutoModelForVision2Seq


# Defining path of the occupancy grid map image
mapImage = "Test_Images/Occupancy_Grid_Map.png"

# Defining output path for saving final image
outputPath = "New_Results/ImageResult.png"

# System requirements setup
# Loading the VLM model
# particularly smolVLM-Instruct 
SOCIAL_MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"

# Initializing the model processor (tokenizer and feature extractor)
socialProcessor = AutoProcessor.from_pretrained(SOCIAL_MODEL_ID)

# Initializing the VLM model and moving it to CPU memory
socialModel = AutoModelForVision2Seq.from_pretrained(SOCIAL_MODEL_ID).to("cpu")

# Setting the model to evaluation mode (disables dropout, etc.)
socialModel.eval()


# This function is used to create a
# social cost map with smolVLM contribution
def Social_Cost_Map(img, output_size):

    # Converting the OpenCV BGR image array to a PIL RGB image object for the VLM
    pilIMG = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    coarse = 20  # Reverting to stable grid size for VLM output (20x20 grid)

    # Shorten the prompt to save tokens 
    # Detailed prompt instructing the VLM to return a specific JSON format
    prompt = f""" Return ONLY a single, clean JSON array of length {coarse},
             where each entry is a list of {coarse} floats in [0,1].
             0=socially acceptable, 1=highly inappropriate.
             """
     
    cleanPromptText = prompt.strip() 
    
    # Formatting the prompt with the required <image> tag
    formatted = f"<image>{prompt}"
    
    # Processing the image and text prompt into model inputs
    inputs = socialProcessor(images=pilIMG, text=formatted, return_tensors="pt").to("cpu")

    # Disabling gradient calculation for inference (saves memory and speeds up)
    with torch.no_grad():
        
        # Generating the VLM response
        output_ids = socialModel.generate(
        
            **inputs,
            max_new_tokens=250,
            do_sample=False,

            # Using temperature 0.0 for deterministic output 
            temperature=0.0 
        )

    # Decoding the generated token IDs back into human-readable text
    text = socialProcessor.decode(output_ids[0], skip_special_tokens=True)
    
    # Aggressive Cleaning is still necessary:
    # Removing the input prompt and formatting tags from the output text
    raw_VLM_Output = text.replace(formatted, "").strip()
    raw_VLM_Output = raw_VLM_Output.replace(cleanPromptText, "").strip() 
    
    # Removing any remaining HTML/XML tags
    clean_VLM_Output = re.sub(r'<[^>]+>', '', raw_VLM_Output)

    # Try to parse JSON substring from the CLEANED output:
    try:
        
        # Attempting a direct JSON load
        data = json.loads(clean_VLM_Output)
        
    except:
        
        # fallback: locate JSON via regex in the CLEANED output
        # Searching for the outermost array structure
        match = re.search(r"\[[\s\S]*\]", clean_VLM_Output)
        
        if not match:
            
            print("No usable JSON from VLM, defaulting to zeros")
            
            # Returning a zero-cost map if parsing fails
            return np.zeros(output_size, dtype=np.float32), raw_VLM_Output 
        
        # Loading JSON from the matched substring
        data = json.loads(match.group(0))

    # Converting the list-of-lists (JSON data) into a NumPy array
    array = np.array(data, dtype=np.float32)

    # Resizing the 20x20 social cost array to the final planner resolution
    arrayResized = cv2.resize(array, output_size, interpolation=cv2.INTER_CUBIC)

    # Normalizing the resized array if it contains non-zero costs
    if arrayResized.max() > 0:
        
        arrayResized /= arrayResized.max()

    # Returning the resized, normalized social cost map and the raw VLM output
    return arrayResized, raw_VLM_Output


# Defining function for evaluating
# path social awareness and optimality
def EvaluatePath(img, start, goal):
    
    # Converting the OpenCV BGR image array to a PIL RGB image object for the VLM
    pilIMG = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Detailed prompt for evaluation
    # Instructions for the VLM to score the path (red line) on key criteria
    prompt = f"""
             Evaluate the path (red line from blue goal to green start) on a scale of
             1 (Poor) to 5 (Excellent) for the following:
             
             1. Social Awareness (Avoids people and seating).
             2. Optimality/Speed (Directness and path length).
             3. Overall Score.
             Return ONLY three comma-separated integers. Do not include your prompt, the criteria names, or any other characters. Example: 4, 5, 4
             """
     
    cleanPromptText = prompt.strip()
    
    # Formatting the prompt with the required <image> tag
    formatted = f"<image>{prompt}"
    
    # Processing the image and text prompt into model inputs
    inputs = socialProcessor(images=pilIMG, text=formatted, return_tensors="pt").to("cpu")

    # Disabling gradient calculation for inference
    with torch.no_grad():
        
        # Generating the VLM response
        output_ids = socialModel.generate(
        
            **inputs,
            max_new_tokens=150,
            do_sample=False,

            # Using temperature 0.0 for deterministic output
            temperature=0.0 
        )

    # Decoding the generated token IDs into text
    evaluationText = socialProcessor.decode(output_ids[0], skip_special_tokens=True)
    
    # Aggressively clean the output
    raw_VLM_Output = evaluationText.replace(formatted, "").strip()
    raw_VLM_Output = raw_VLM_Output.replace(cleanPromptText, "").strip()
    clean_VLM_Output = re.sub(r'<[^>]+>', '', raw_VLM_Output).strip()
    
    # Try to parse the scores
    scores = None
 
    try:
 
        # Look for 3 comma-separated numbers (allowing for some whitespace)
        match = re.search(r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", clean_VLM_Output)
 
        if match:
 
            # Extracting the three scores as integers
            scores = [int(match.group(i)) for i in range(1, 4)]
 
    except:
 
        # Fall through to the final return if parsing fails
        pass

    # Format the final output for the user
    if scores and len(scores) == 3:
 
        # Returning structured evaluation scores
        return f"""
               1. Social Awareness: {scores[0]} / 5
               2. Optimality/Speed: {scores[1]} / 5
               3. Overall Score: {scores[2]} / 5
               Raw VLM Output (cleaned): {clean_VLM_Output}
               """
    else:
         
        # Return the raw cleaned output if the pattern wasn't found
        return f"VLM failed to return structured scores. Raw cleaned output:\n{clean_VLM_Output}"

# Reference coordinates (defined in 500x500 space) 
# Reference width for scaling coordinates
refWidth = 500

# Reference height for scaling coordinates
refHeight = 500

# Defining the starting point coordinate in the 500x500 reference space
refStart = [485, 250]   

# Defining the goal point coordinate in the 500x500 reference space
refGoal  = [15, 440]     

# Dynamic: Reading actual image to get real size 
# Temporarily reading the map image to check its actual dimensions
tempImage = cv2.imread(mapImage)

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

    # Defining movement directions (dx, dy, base_cost) for 8 directions
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

                if grid[ny, nx] == 1:

                    # Initializing turn penalty
                    turnCost = 0

                    # Calculating turn penalty: checks if the move direction changed
                    if prev is not None:

                        # Compares (current - prev) vector to (next - current) vector
                        if (current[0] - prev[0], current[1] - prev[1]) != (nx - x, ny - y):

                            turnCost = turnPenalty

                    # Initializing curvature penalty
                    curvatureCost = 0

                    # Calculating curvature penalty if applicable (needs 3 points: prevPrev, prev, current)
                    if prev is not None and prevPrev is not None:

                        # Computing vector from prevPrev to prev
                        v1 = np.array([prev[0] - prevPrev[0], prev[1] - prevPrev[1]])

                        # Computing vector from current to next
                        v2 = np.array([nx - x, ny - y])

                        # Calculating angle if vectors are non-zero
                        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:

                            # Computing cosine of angle between the two movement segments
                            cosAngle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

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
            
            x += xInc
            error -= dy
            
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

            # Move one step backward if line of sight fails
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
    uNew = np.linspace(0, 1, len(path) * 4)

    # Evaluating spline at dense parameter points
    out = splev(uNew, tck)

    # Converting to integer points and zipping X and Y coordinates
    smoothed = list(zip(out[0].astype(int), out[1].astype(int)))

    # Returning smoothed path
    return smoothed


# Defining main function for running planner
def RunPlanner():

    # Reading map image
    image = cv2.imread(mapImage)

    # Resize to planner size (test or original)
    image = cv2.resize(image, (plannerWidth, plannerHeight))

    # Checking if image is loaded
    if image is None:

        print("Map not found")

        return

    # Copying image for drawing the path and markers
    imgDraw = image.copy()

    # Loading YOLO model for object detection
    model = YOLO(yoloModelPath)

    # Detecting obstacles
    # Run YOLO on fixed 320x320 for stable detection
    yoloSize = 320
    
    # Resizing the map image to YOLO's input size
    yoloImage = cv2.resize(image, (yoloSize, yoloSize))
    
    # Running detection
    results = model(yoloImage, save=False)

    # Initializing list of bounding boxes
    boxes = []
    
    # Setting a minimum size for detected boxes, scaled
    minBoxSize = max(8, int(25 * scaleX))  # Stronger min size

    # Processing YOLO detection results
    for r in results:
        
        # Iterating over all detected bounding boxes
        for box in r.boxes.xyxy.cpu().numpy():
            
            # Unpacking coordinates (x_min, y_min, x_max, y_max)
            x1, y1, x2, y2 = box.astype(float)
            
            # Rescaling X coordinates from YOLO size (320) to planner size
            x1 = x1 * plannerWidth  / yoloSize
            x2 = x2 * plannerWidth  / yoloSize
            
            # Rescaling Y coordinates from YOLO size (320) to planner size
            y1 = y1 * plannerHeight / yoloSize
            y2 = y2 * plannerHeight / yoloSize

            # Converting coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Calculating width and height
            w, h = x2 - x1, y2 - y1

            # Enforcing minimum size on width
            if w < minBoxSize:
                
                center = (x1 + x2) // 2
                
                x1 = center - minBoxSize // 2
                x2 = center + (minBoxSize + 1) // 2
                
            # Enforcing minimum size on height
            if h < minBoxSize:
                
                center = (y1 + y2) // 2
                
                y1 = center - minBoxSize // 2
                y2 = center + (minBoxSize + 1) // 2

            # Ensuring minimum 1-pixel size
            if x2 <= x1: x2 = x1 + 1
            
            if y2 <= y1: y2 = y1 + 1

            # Clipping coordinates to be within image bounds
            x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, [plannerWidth-1, plannerHeight-1]*2)
            
            # Appending final box coordinates
            boxes.append((int(x1), int(y1), int(x2), int(y2)))

    # Filling grid
    # Initializing obstacle grid (0: free, 1: obstacle)
    grid = np.zeros((plannerHeight, plannerWidth), dtype=np.uint8)
    
    # Filling the grid with detected bounding boxes (set to 1)
    for x1, y1, x2, y2 in boxes:
        
        grid[y1:y2, x1:x2] = 1

    # Small dilation to slightly expand detected obstacles
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
    inflated = cv2.dilate(grid, kernel, iterations=1)

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

    # Protect start
    startX, startY = start
    
    # Creating a circular mask around the start point
    startMask = (xGrid - startX)**2 + (yGrid - startY)**2 <= protectionRadius**2
    
    # Setting masked areas in planner grid to free (1)
    planner[startMask] = 1
    
    # Setting masked areas in inflated grid to non-obstacle (0)
    inflated[startMask] = 0

    # Protect goal
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
    
    # Inverse-distance cost function (high cost near obstacles, low far away)
    costMap = costAmplifier * (1.0 / (distance + 1.0))**3
    
    # Cap for stability
    costMap = np.clip(costMap, 0, 100)  

    print("Querying VLM for social cost...")
    
    # Getting social cost map from VLM and its raw output
    socialCost, vlmReasoning = Social_Cost_Map(image, (plannerWidth, plannerHeight))
    
    # Weight for the social cost component
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

        x, y = points
        attempts = 0

        # Looping until free cell is found or attempts exceed 100
        while planner[y, x] == 0 and attempts < 100:

            # Shifting X towards the nearest image half
            if x < plannerWidth / 2:

                x += 1

            else:

                x -= 1

            # Shifting Y towards the nearest image half
            if y < plannerHeight / 2:

                y += 1

            else:

                y -= 1

            # Ensuring the shifted point is still inside wall margins
            x, y = ShiftInsideWall([x, y])

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
  
        print("No path found")

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
    for x1, y1, x2, y2 in boxes:

        # Drawing the yellow bounding box outline
        cv2.rectangle(imgDraw, (x1, y1), (x2, y2), (0, 255, 255), boxThickness)

    # Saving output image
    cv2.imwrite(outputPath, imgDraw)

    # Printing saved message
    print("Saved", outputPath)

    # VLM Reasoning and Evaluation 
    print("\nVLM SOCIAL COST REASONING")
    print(vlmReasoning)

    print("\nVLM PATH EVALUATION")
    
    # Must use the final image with path drawn
    # Executing VLM path evaluation function
    evaluationResult = EvaluatePath(imgDraw, start, goal)
    print(evaluationResult)

    # Visualizing the social cost map
    visualization = (socialCost * 255).astype(np.uint8)
    cv2.imshow("Social Cost", visualization)

    # Showing A* planner window 
    cv2.imshow("A* Planner", imgDraw)

    # Waiting key press to close windows
    cv2.waitKey(0)

    # Closing all windows
    cv2.destroyAllWindows()

# Running A* planner function
if __name__ == "__main__":
    
    RunPlanner()