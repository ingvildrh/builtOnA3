import numpy as np
import cv2
import json
import os
from PIL import Image

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


for key, value in data.items():
    filename = value["filename"]

    img_path = img_dir + "/" + filename
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w, _= img.shape
    
    mask = np.zeros((h,w))

    cv2.imwrite("mask_dir/", mask)