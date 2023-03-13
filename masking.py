import os
import cv2 as cv2
import json
import numpy as np 

source_folder ="C:\\Users\\ingvilrh\\OneDrive - NTNU\\Masteroppgave23\\eyeDetection\\images"
json_path = "C:\\Users\\ingvilrh\\OneDrive - NTNU\\Masteroppgave23\\eyeDetection\\via_project_13Mar2023_11h0m_json.json"

#json_path = "C:\Users\ingvilrh\OneDrive - NTNU\Masteroppgave23\eyeDetection\via_project_13Mar2023_10h20m_json.json"                     # Relative to root directory
count = 0                                           # Count of total images saved
file_bbs = {}                                       # Dictionary containing polygon coordinates for mask
MASK_WIDTH = 256				    # Dimensions should match those of ground truth image
MASK_HEIGHT = 256									

# Read JSON file
with open(json_path) as f:
  data = json.load(f)

# Extract X and Y coordinates if available and update dictionary
def add_to_dict(data, itr, key, count):
    #try:
    x_points = data[itr]["regions"][count]["shape_attributes"]["x"]
    #print(x_points)
    y_points = data[itr]["regions"][count]["shape_attributes"]["y"]

    width = data[itr]["regions"][count]["shape_attributes"]["width"]
    height = data[itr]["regions"][count]["shape_attributes"]["height"]
    #print(y_points)
    # except:
    #     print("No BB. Skipping", key)
    #     return
    
    all_points = []
    #for i, x in enumerate(x_points):
    all_points.append([x_points, y_points, width, height])
    
    file_bbs[key] = all_points
    print(file_bbs)
  
for itr in data:
    file_name_json = data[itr]["filename"]
    print(itr)
    sub_count = 0 # Contains count of masks for a single ground truth image
    
    if len(data[itr]["regions"]) > 1:
        for _ in range(len(data[itr]["regions"])):
            key = file_name_json[:-4] + "*" + str(sub_count+1)
            add_to_dict(data, itr, key, sub_count)
            sub_count += 1
    else:
        add_to_dict(data, itr, file_name_json[:-4], 0)

			
print("\nDict size: ", len(file_bbs))

for file_name in os.listdir(source_folder):
    print(file_name)
    to_save_folder = os.path.join(source_folder, file_name[:-4])
    image_folder = "C:\\Users\\ingvilrh\\OneDrive - NTNU\\Masteroppgave23\\eyeDetection\\images"
    mask_folder = "C:\\Users\\ingvilrh\\OneDrive - NTNU\\Masteroppgave23\\eyeDetection\\masks"
    curr_img = os.path.join(source_folder, file_name)
    
    # make folders and copy image to new location
    # os.mkdir(to_save_folder)
    # os.mkdir(image_folder)
    # os.mkdir(mask_folder)
    # os.rename(curr_img, os.path.join(image_folder, file_name))
        
# For each entry in dictionary, generate mask and save in correponding 
# folder
for itr in file_bbs:

    mask_folder = "C:\\Users\\ingvilrh\\OneDrive - NTNU\\Masteroppgave23\\eyeDetection\\masks"
    mask = np.zeros((MASK_WIDTH, MASK_HEIGHT))
    try:
        arr = np.array(file_bbs[itr]) #storing x,y,width,height
        print(arr)
    except:
        print("Not found:", itr)
        continue
    count += 1
    x = arr[0][0]
    y = arr[0][1]
    width = arr[0][2]
    height = arr[0][3]
    print(itr)
    cv2.rectangle(mask, (x, y), (x + width, y + height), (255, 0, 0), -1)
    window = "Filles Rectangle"
    cv2.imshow("Filled Rectangle", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # if len(num_masks) > 1:
    # 	cv2.imwrite(os.path.join(mask_folder, itr.replace("*", "_") + ".png") , mask)    
    # else:
    #     cv2.imwrite(os.path.join(mask_folder, itr + ".png") , mask)
        
print("Images saved:", count)