#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from os import walk
import os
from PIL import Image
 
from math import sqrt, floor

parser = argparse.ArgumentParser(description='')

parser.add_argument('--img_in', type=str, help='Dirección de la carpeta con las imágenes.')

args = parser.parse_args()

res = 256

def load_imgs(filenames):
    
    # Separa imgs y máscaras en listas diferentes
    img_names = [ args.img_in + '/' + img for img in filenames if 'image' in img ]
    img_names.sort()
    mask_names = [ args.img_in + '/' +  mask for mask in filenames if 'mask' in mask ]
    mask_names.sort()
    imgs = [ Image.open(img) for img in img_names  ]
    masks = [ Image.open(mask) for mask in mask_names ]
    return imgs, masks

#Código para obtener cuadrado perfecto más cercano tomado de https://www.geeksforgeeks.org/closest-perfect-square-and-its-distance/
def get_size(ex_num, res):
    x = floor(sqrt(ex_num))
     
    # Checking if N is itself a perfect square
    if (sqrt(ex_num) - floor(sqrt(ex_num)) == 0):
        print(ex_num,0)
        return
 
    # Variables to store first perfect
    # square number above and below N
    aboveN = (x+1)*(x+1)
    belowN = x*x
 
    # Variables to store the differences
    diff1 = aboveN - ex_num
    diff2 = ex_num - belowN
 
    if (diff1 > diff2):
        return (2*sqrt(belowN)*res,res*( 1+sqrt(belowN)) )
    else:
        return ( 2*sqrt(aboveN)*res, sqrt(aboveN)*res )
    
def create_mosaic(filenames, res):
    imgs, masks = load_imgs(filenames)
    
    dims = get_size(len(filenames)/2, res)
    
    dims = (int(dims[0]), int(dims[1]))
    
    dst = Image.new(mode='RGB', size=dims, color=(255,255,255))
    
    x_counter = 0
    y_counter = 0
    
    
    for i in range( int(dims[1]/res) ):
        for i in range( min( int(dims[0]/(res*2) ), len(imgs) ) ):
            dst.paste(imgs[-1], (x_counter*res, y_counter*res))
            dst.paste(masks[-1], ((x_counter+1)*res, y_counter*res))
            
            imgs.pop()
            masks.pop()
            
            x_counter += 2
        y_counter += 1
        x_counter = 0
    
        
    return dst        


# get file names
filenames = next(walk(args.img_in), (None, None, []))[2]  # [] if no file
filenames.sort()

create_mosaic(filenames, res).save(args.img_in + '/' + 'mosaic.png')




 
 

    





