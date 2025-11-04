# Author: Endri Dibra 
# Project: ARP mini Project 

# Importing the required libraries
import cv2


# Reading the image 
image = cv2.imread("Test_Images/Occupancy_Grid_Map.png")

# Resizing the image to 500 x 500
image = cv2.resize(image, (500, 500))

# Displaying the image
cv2.imshow("The Map", image)

# Waiting for key interaption
cv2.waitKey(0)

# Terminating all opened windows
cv2.destroyAllWindows()