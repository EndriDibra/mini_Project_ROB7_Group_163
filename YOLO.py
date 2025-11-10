# Author: Endri Dibra 
# Project: ARP mini Project 

# Importing the required libraries
import cv2
from ultralytics import YOLO


# Reading the image
image = cv2.imread("Test_Images/Occupancy_Grid_Map.png")

# Resizing the image to 500 x 500
image = cv2.resize(image, (500, 500))

# Loading YOLOv11n model for object detection
model = YOLO("yolo11n.pt")

# Applying the model on the image to detect objects
results = model(image, save=False)
results = results[0].plot()

# Displaying the image with the detected obstacles
cv2.imshow("YOLO Detection", results)

# Waiting until q key is pressed
cv2.waitKey(0)

# Terminating all opened windows
cv2.destroyAllWindows() 