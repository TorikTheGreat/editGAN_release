#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:33:50 2022

@author: marco
"""

import argparse
import numpy as np
from PIL import Image
from os import walk
import os
from random import sample
import shutil


parser = argparse.ArgumentParser(description='')

parser.add_argument('--img_in', type=str, help='Dirección de la carpeta con las imágenes.')
parser.add_argument('--mask_in', type=str, help='Dirección de la carpeta con las máscaras.')
parser.add_argument('--img_out', type=str, help='Carpeta donde copiar las imágenes' )
parser.add_argument('--sample_size', type=int, help='Cantidad de imágenes que copiar'  )

args = parser.parse_args()


# get file names
filenames = next(walk(args.img_in), (None, None, []))[2]  # [] if no file
filenames.sort()

masknames = next(walk(args.mask_in), (None, None, []))[2]  # [] if no file
masknames.sort()

if args.sample_size <= len(filenames):
    
    #Elige al azar las imágenes y máscaras correspondientes
    filenames = sample(filenames,args.sample_size)
    masknames = [ mask for mask in masknames if mask[:-14]+mask[-9:] in filenames]
    filenames.sort()
    masknames.sort()
    
    
else:
    print('sample_size demasiado grande para conjunto de datos.')
    
for img, msk in zip(filenames, masknames):
    
    # Mueve la imagen
    shutil.move(args.img_in + '/' + img, args.img_out + '/' + img )
    print(f'Moved {img} from {args.img_in} to {args.img_out}')
    
    # Mueve la máscara
    shutil.move(args.mask_in + '/' + msk, args.img_out + '/' + msk )
    print(f'Moved {msk} from {args.mask_in} to {args.img_out}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    