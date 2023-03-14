import numpy as np
import cv2
import json
import os
from PIL import Image

'''
This file reads a JSON file with annotations of desired regions created in https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html
The file is saved as a JSON file which can be properly shown in a JSON viewer http://jsonviewer.stack.hu/

'''
#Reading JSON file
f = open('tailDetection.json')
data = json.load(f)

data = data["_via_img_metadata"]

img_dir = "C:/Users/ingvilrh/OneDrive - NTNU/Masteroppgave23/eyeDetection/images/"

msk_dir = "C:/Users/ingvilrh/OneDrive - NTNU/Masteroppgave23/eyeDetection/masks/"

#Loop over the data and save the masks
# for file_name in os.listdir(img_dir):
#     image = Image.open(img_dir+"/"+file_name)
#     image.show()
#     print(file_name)

#Loop over the data and save masks
for key, value in data.items():
    filename = value["filename"]

    img_path = img_dir + "/" + filename
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w, _= img.shape
    
    mask = np.zeros((h,w))

    regions = value["regions"]
    for region in regions:
        shape_attribues = region["shape_attributes"]
       
        x_points = shape_attribues["all_points_x"]
        y_points = shape_attribues["all_points_y"]

        contours = []
        for x,y in zip(x_points, y_points):
            contours.append((x,y))
        contours = np.array(contours)
        print(contours)

        cv2.drawContours(mask, [contours],-1, 255, -1) #if last -1 is changed, the number is a line width

    cv2.imwrite(msk_dir+filename, mask)
        
        



    cv2.imwrite("mask_dir.png", mask)