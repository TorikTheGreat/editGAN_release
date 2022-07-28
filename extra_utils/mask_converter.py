#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 20:45:25 2022

@author: marco
"""
import argparse
import numpy as np
from PIL import Image
from os import walk


parser = argparse.ArgumentParser(description='Generador de máscaras en el formato que ocupa editgan. Color blanco: nematodo, color negro: fondo')

parser.add_argument('mask_in', type=str, help='Dirección de la carpeta con las imágenes de las mascaras.')

parser.add_argument('mask_out', type=str, help='Dirección de la carpeta donde guardar las máscaras nuevas.')


args = parser.parse_args()

# get file names
filenames = next(walk(args.mask_in), (None, None, []))[2]  # [] if no file
print(filenames)



for i in filenames:
    path = args.mask_in + '/' + i
    img = Image.open(path)
    img_npy = np.asarray(img)[:,:,0]/255
    new_name ='image_mask' + i[6:-4] + '.npy' 
    
    np.save(args.mask_out + '/' + new_name , img_npy)
    print(f'Mask {i} saved as {new_name}')
    
