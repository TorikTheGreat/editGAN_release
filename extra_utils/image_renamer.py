#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:37:36 2022

@author: marco
"""
import argparse
import numpy as np
from PIL import Image
from os import walk
import os


parser = argparse.ArgumentParser(description='Cambiador de nombres de las imagenes sinteticas al formato image_#.png')

parser.add_argument('img_in', type=str, help='Dirección de la carpeta con las imágenes.')

args = parser.parse_args()

# get file names
filenames = next(walk(args.img_in), (None, None, []))[2]  # [] if no file
print(filenames)

counter = 0

for i in filenames:
    
    
    name = 'image_' + str(counter) + '.png'
    
    os.rename(args.img_in + '/' + i, args.img_in + '/' + name )
    
    print(f'Renamed {i} to {name} \n')
    
    counter += 1
    