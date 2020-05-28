# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
from os import listdir
from os.path import isfile, join


info = [
    {
        "name": "V8 Splash - Strawberry Kiwi",
        "calories": "84cal / 200ml"
    },
    {
        "name": "Del Valle del Prado - Néctar Melocotón",
        "calories": "120kcal / 200ml"
    }

]
threshold = 200

def ORB_detector(new_image):
    # Function that compares input image to template
    # It then returns the number of ORB matches between them
    
    grey = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Create ORB detector with 1000 keypoints with a scaling pyramid factor of 1.2
    orb = cv2.ORB_create(nfeatures = 1000)

    # Detect keypoints of original image
    (kp1, des1) = orb.detectAndCompute(grey, None)

    data_path = './images/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    index = 0
    matches_len = 0
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        image_template = cv2.imread(image_path, 0)
        # Detect keypoints of rotated image
        (kp2, des2) = orb.detectAndCompute(image_template, None)
        # Create matcher 
        # Note we're no longer using Flannbased matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
        # Do matching
        matches = bf.match(des1,des2)
        # Sort the matches based on distance.  Least distance
        # is better
        matches = sorted(matches, key=lambda val: val.distance)
        if(len(matches) > threshold):
            matches_len = len(matches)
            index = i
            break
    if(matches_len > 0):
        return matches_len, index
    else:
        return 0, -1

cap = cv2.VideoCapture(-1)

# Load our image template, this is our reference image
#image_template = cv2.imread('images/1.jpg', 0) 
# image_template = cv2.imread('images/kitkat.jpg', 0) 

while True:

    # Get webcam images
    ret, frame = cap.read()
    # Get height and width of webcam frame
    height, width = frame.shape[:2]

    # Define ROI Box Dimensions (Note some of these things should be outside the loop)
    top_left_x = math.floor(width / 3)
    top_left_y = math.floor((height / 2) + (height / 4))
    bottom_right_x = math.floor((width / 3) * 2)
    bottom_right_y = math.floor((height / 2) - (height / 4))
    start_point = (top_left_x,top_left_y)
    end_point =(bottom_right_x,bottom_right_y)
    color = (255, 0, 0)
    
    # Draw rectangular window for our region of interest
    cv2.rectangle(frame, start_point, end_point, color, 3)
    
    # Crop window of observation we defined above
    cropped = frame[bottom_right_y:top_left_y , top_left_x:bottom_right_x]

    # Flip frame orientation horizontally
    frame = cv2.flip(frame,1)
    
    # Get number of ORB matches 
    matches, index = ORB_detector(cropped)
    
    #print("{0} | {1}".format(matches, index))
    
    # Display status string showing the current no. of matches 
    output_string = "Matches = " + str(matches)
    cv2.putText(frame, output_string, (50,450), cv2.FONT_HERSHEY_COMPLEX, 2, (250,0,150), 2)
    
    
    # Our threshold to indicate object deteciton
    # For new images or lightening conditions you may need to experiment a bit 
    # Note: The ORB detector to get the top 1000 matches, 350 is essentially a min 35% match
    #threshold = 200
    
    # If matches exceed our threshold then object has been detected
    if matches > threshold:
        output_string = info[index]["name"] + " " + info[index]["calories"]
        cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), (0,255,0), 3)
        cv2.putText(frame,output_string,(50,50), cv2.FONT_HERSHEY_COMPLEX, 0.5 ,(255,255,255), 1)
    
    cv2.imshow('Object Detector using ORB', frame)
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()  