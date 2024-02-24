import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sklearn as sk

"""
DRAWING A CIRCLE AROUND THE VESSEL
"""

def draw_circle(image):

    # Convert the image to grayscale
    cimg = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    # Apply Hough Circle Transform -> find the circles in the colored image
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 
                            4, 10000, param1=200, param2=200, 
                            minRadius=1500, maxRadius=1800)

    #round the values to whole numbers and make them of type int
    circles = np.uint16(np.around(circles)) 

    for c in circles[0, :]:
        #draw the outer circle on the greyscale image
        cv.circle(cimg, (c[0], c[1]), c[2], (0, 255, 0), 5) 

    return cimg

"""
CROPPING TO THE GREEN CIRCLE
"""

def crop_to_circle(image):

    # Convert the image to HSV color space
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Define the range of colors for the petri dish (green color)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get only green colors (the petri dish)
    mask = cv.inRange(hsv, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to remove small noise
    min_area = 1000  # Adjust as needed
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]

    # Crop and save bounding boxes
    for i, contour in enumerate(contours):
        x, y, w, h = cv.boundingRect(contour)
        
        # Crop the bounding box region
        cropped_image = image[y:y+h, x:x+w]

    return cropped_image

"""
CUTTING OUT OUTSIDE OF THE CIRCLE
"""

def mask_out_circle(image):
    # Get image dimensions
    height, width = image.shape[:2]

    # Create a black image with alpha channel
    mask = np.zeros((height, width, 4), dtype=np.uint8)

    # Calculate circle parameters
    center = (width // 2, height // 2)
    radius = min(center[0], center[1])-10

    # Draw the circle on the mask
    cv.circle(mask, center, radius, (255, 255, 255, 255), -1)

    # Set the alpha channel of the input image to 0 outside the circle
    image_with_alpha = np.concatenate((image, np.full((height, width, 1), 255, dtype=np.uint8)), axis=-1)
    image_with_alpha[mask[:, :, 3] == 0] = [0, 0, 0, 0]

    return image_with_alpha

def preprocess_image(image):
    # Apply the functions
    image = draw_circle(image)
    image = crop_to_circle(image)
    image = mask_out_circle(image)

    return image

def main ():

    # iterate through images in the train_data folder, apply preprocess_image function and save the images in the preprocessed_data folder
    for i in range(1, 17418):

        # try to open the image, if it fails, continue to the next image
        try:
            img = cv.imread(f"data/train_data/{i}.jpg", 0)
            img = preprocess_image(img)
            cv.imwrite(f"data/preprocessed_data/{i}.jpg", img)

        except:
            continue

if __name__ == "__main__":
    main()