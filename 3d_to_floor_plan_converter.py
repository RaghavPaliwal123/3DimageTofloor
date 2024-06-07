import cv2
import numpy as np

def convert_to_floor_plan(image_path, threshold=200):
    # Read the input 3D image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply thresholding to identify floor areas
    _, floor_mask = cv2.threshold(image, threshold, 250, cv2.THRESH_BINARY)
    
    # Find contours of the floor areas
    contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank canvas for the floor plan
    floor_plan = np.zeros_like(image)
    
    # Draw contours on the floor plan
    cv2.drawContours(floor_plan, contours, -1, (250), thickness=cv2.FILLED)
    
    return floor_plan

# Example usage:
input_image_path = "red-aesthetic-art-john-wick-hd-6ia5zmm6kvrchebp.webp"
floor_plan = convert_to_floor_plan(input_image_path)
cv2.imshow("Floor Plan", floor_plan)
cv2.waitKey(0)
cv2.destroyAllWindows()
