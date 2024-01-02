# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:51:50 2023

@author: parth
"""

from PIL import Image
import os

# Set the path to your image folder
input_folder = "P:\Dataset\Merged_Dataset"
output_folder = "P:\Dataset\Merged_Dataset_Processed"

# Set the target size
target_size = (48, 48)

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through each image in the input folder
for filename in os.listdir(input_folder):
    # Load the image
    input_path = os.path.join(input_folder, filename)
    img = Image.open(input_path)

    # Resize the image
    resized_img = img.resize(target_size)

    # Save the resized image to the output folder
    output_path = os.path.join(output_folder, filename)
    resized_img.save(output_path)

print("Resizing complete.")