# Author: Endri Dibra 
# Project: ARP mini Project 


# Importing the required libraries
import cv2 

# System requirements setup 
# The Occupancy grid map
# image to be used for the process
IMAGE_PATH = "Test_Images/Occupancy_Grid_Map.png"

# Output image path
OUTPUT_PATH = "New_Results/ImageResultp.png"

# New size of the occupancy grid map image
# for computing resourse management
RESIZE = (500, 500) 

# Goal coordinates
GOAL_X = 250 
GOAL_Y = 480 

# Dot characteristics
DOT_COLOR = (0, 0, 255) 
DOT_RADIUS = 8
DOT_THICKNESS = -1 

print(f"Loading image: {IMAGE_PATH}...")

# Loading image
image = cv2.imread(IMAGE_PATH)

# Checking if image exists
if image is None:
    
    print(f"Error! Image not found.")
    
    raise FileNotFoundError(f"{IMAGE_PATH} not found")

# Resizing image
if RESIZE is not None:
    
    image = cv2.resize(image, RESIZE)

print("Image loaded successfully.")

# Drawing the circle
cv2.circle(image, (GOAL_X, GOAL_Y), DOT_RADIUS, DOT_COLOR, DOT_THICKNESS)

# Adding text label
cv2.putText(image, "Goal", (GOAL_X + 10, GOAL_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DOT_COLOR, 2)

# Displaying output text
cv2.imshow("Image with Goal", image)
cv2.imwrite(OUTPUT_PATH, image)

print(f"Saved image with goal to {OUTPUT_PATH}")

# Terminating window by pressing a key
cv2.waitKey(0)

# Terminating all opened windows
cv2.destroyAllWindows()