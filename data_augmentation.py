from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image
#from torchsummary import summary
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale

import os
import utils
import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torchvision
from torch import optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from imgaug import augmenters as iaa
import random
import matplotlib.image as mpimg
import cv2



# Function to rename multiple files
def renamer(path):
    iterater = 0     
    for filename in os.listdir(path):
        print(str(iterater))
        dst = str(iterater) + ".jpg"
        src = os.path.join(path, filename)  # add trailing slash or backslash to path
        dst = os.path.join(path, dst)  # add trailing slash or backslash to path
        os.rename(src, dst) 
        iterater += 1

#Images verification method
def verifier(path):    
    for filename in os.listdir(path):
        try:
            img = Image.open(os.path.join(path, filename)) # add trailing slash or backslash to path
        except (Exception, FileNotFoundError, AttributeError):
            os.remove(os.path.join(path, filename)) # add trailing slash or backslash to path
        try:
            img.verify()
        except (Exception, FileNotFoundError, AttributeError):
            os.remove(os.path.join(path, filename)) # add trailing slash or backslash to path

# Extra images generating loop
def img_generator(path, no_gen):
  i = 0
  for filename in os.listdir(path):
    #try:
    i +=1
    img = load_image(path + '/' + filename)
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    save_image(augment(img), path + '/Generated'+ str(i) +'.jpg')
    #except (ValueError, OSError):
    #  print("in exception")
    #  pass

def load_image(infilename):
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image(npdata, outfilename) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save(outfilename, "JPEG")

#Generating extra preprocessed images
#This could be edited to choose the augmentation we want
def augment(img):
  par_1 = random.uniform(0.1, 1.0)
  par_2 = random.uniform(1.0, 15.0)
  par_3 = random.uniform(2.0, 40.0)
  par_4 = random.uniform(0.1, 1.0)
  par_5 = random.uniform(0.1, 1.0)
  par_6 = random.uniform(0.01, 0.2)
  affine = iaa.Affine(rotate=(-10, 10), mode = 'edge')
  img = affine.augment_image(img)
  #blurer = iaa.GaussianBlur(iaa.Uniform(0.1,par_1)) 
  #img = blurer.augment_image(img)
  elastic = iaa.ElasticTransformation(sigma=par_2, alpha=par_3)
  img=elastic.augment_image(img)
  flp=iaa.Flipud(p=par_4)
  img=flp.augment_image(img)
  salt = iaa.SaltAndPepper(p=par_6)
  img=salt.augment_image(img)
  flp2=iaa.Fliplr(p=par_5)
  img=flp2.augment_image(img)
  crop = iaa.CropToFixedSize(1500,1300,position="center-bottom")
  img = crop.augment_image(img)
  return img

def augment2(img):
    brightness = iaa.WithBrightnessChannels(iaa.Add((-50, 50)))
    img = brightness.augment_image(img)
    aug = iaa.WithHueAndSaturation([
    iaa.WithChannels(0, iaa.Add((-30, 10))),
    iaa.WithChannels(1, [
        iaa.Multiply((0.5, 1.5)),
        iaa.LinearContrast((0.75, 1.25))])])
    img = aug.augment_image(img)
    hueSaturation = iaa.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5))
    img = hueSaturation.augment_image(img)
    return img


def augment_all_classes(PATH):
    for class_folder in os.listdir(PATH):
        class_path = PATH + '/' + class_folder
        num_img = len(os.listdir(class_path))
        img_generator(class_path, num_img-1)

def remove_augmented_img(PATH): 
    for class_folder in os.listdir(PATH):
            class_path = PATH + '/' + class_folder
            for img_name in os.listdir(class_path):
                # Check if the file is an image file
                if ('Generated' in img_name):
                    # Delete the image file using os.remove()
                    image_path = class_path + "/" + img_name
                    os.remove(image_path)

def plot_images(PATH, class_name, top=25):
            class_path = PATH + '/' + class_name
            plt.figure(figsize = (12,12))
            num_img = 0
            for img_name in os.listdir(class_path):
                if num_img >= top:
                    break
                num_img += 1
                image_path = os.path.join(class_path, img_name)
                plt.subplot(5, 5, num_img)
                img = mpimg.imread(image_path)
                plt.imshow(img)
            plt.tight_layout()
            plt.show()

def main():
    augment_all_classes('C:/Users/ingvilrh/OneDrive - NTNU/Masteroppgave23/full_fishdata')


if __name__ == "__main__":
    main()