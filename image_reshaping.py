# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:41:18 2023

@author: Hritik
"""

from PIL import Image
import numpy as np
import sys
import os
import csv

# default format can be changed as needed
def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    labels = []
    names = []
    keywords = {"L": "0", "S": "0.5", "R": "1"}  # keys and values to be changed as needed
    for root, dirs, files in os.walk(myDir, topdown=True):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
                for keyword in keywords:
                    if keyword in name:
                        labels.append(keywords[keyword])
                    else:
                        continue
                names.append(name)
    return fileList, labels, names

# load the original image
myFileList, labels, names = createFileList('P:\Dataset\Merged_Dataset_Processed')
i = 0
for file in myFileList:
    print(file)
    img_file = Image.open(file)
    # img_file.show()
    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode
    # Make image Greyscale
    img_grey = img_file.convert('RGB')
    # img_grey.save('result.png')
    # img_grey.show()
    # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=int).reshape((width, height, 3))
    value = value.flatten()

    value = np.append(value, labels[i])
    i += 1

    print(value)
    with open("P:\Dataset\RGBdirections.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)
