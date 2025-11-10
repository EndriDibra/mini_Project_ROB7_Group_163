# Author: Endri Dibra 
# Project: ARP mini Project 


# Importing the required libraries
import cv2 

# System requirements setup 
# The Occupancy grid map
# image to be used for the process
IMAGE_PATH = "Test_Images/Playground.jpg"

# Output image path
OUTPUT_PATH = "Points_Images/ImageResultPlayground.png"

# New size of the occupancy grid map image
# for computing resourse management
RESIZE = (500, 500) 

# Start coordinates
START_X = 40 
START_Y = 480 

# Goal coordinates
GOAL_X = 240 
GOAL_Y = 330 

# Dot characteristics for start point
S_DOT_COLOR = (0, 255, 0) 
S_DOT_RADIUS = 6
S_DOT_THICKNESS = -1

# Dot characteristics for goal point
G_DOT_COLOR = (255, 0, 0) 
G_DOT_RADIUS = 6
G_DOT_THICKNESS = -1 

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

# Drawing the start circle
cv2.circle(image, (START_X, START_Y), S_DOT_RADIUS, S_DOT_COLOR, S_DOT_THICKNESS)

# Adding text label
cv2.putText(image, "Start", (START_X + 10, START_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, S_DOT_COLOR, 2)

# Drawing the goal circle
cv2.circle(image, (GOAL_X, GOAL_Y), G_DOT_RADIUS, G_DOT_COLOR, G_DOT_THICKNESS)

# Adding text label
cv2.putText(image, "Goal", (GOAL_X + 10, GOAL_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, G_DOT_COLOR, 2)

# Displaying output text
cv2.imshow("Image with Start and Goal points", image)
cv2.imwrite(OUTPUT_PATH, image)

# Saving output image
print(f"Saved image with goal to {OUTPUT_PATH}")

# Terminating window by pressing a key
cv2.waitKey(0)

# Terminating all opened windows
cv2.destroyAllWindows()
