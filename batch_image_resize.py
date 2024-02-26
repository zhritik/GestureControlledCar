# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:51:50 2023

@author: Hritik
"""

from PIL import Image
import os

#setting path to image folders
input_folder = "P:\Dataset\Merged_Dataset"
output_folder = "P:\Dataset\Merged_Dataset_Processed"

#target size of images
target_size = (48, 48)

os.makedirs(output_folder, exist_ok=True)

#loop and resize
for filename in os.listdir(input_folder):
    #loading OG image
    input_path = os.path.join(input_folder, filename)
    img = Image.open(input_path)

    #resizing the image
    resized_img = img.resize(target_size)

    #store resized image
    output_path = os.path.join(output_folder, filename)
    resized_img.save(output_path)

print("Resizing complete.")
