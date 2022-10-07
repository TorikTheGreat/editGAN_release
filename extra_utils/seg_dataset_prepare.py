#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Este script se ocupa de reformatear las máscaras de manera que el valor de los 
pixeles nematodos sea 1. Esto porque algunos modelos de segmentación, como los que utiliza
mmsegmentation, sugieren utilizar máscaras en modo P (paleta) de manera que cada color, o
clase, tenga un número asociado. En este caso hay solo una clase, entonces tiene el número 1.

Adicionalmente, hace un split de train/val para entrenar con los datos a un segmentador.

"""

from PIL import Image
import numpy as np
import argparse
from os import walk
from sklearn.model_selection import train_test_split
import shutil
import os
import concurrent.futures
from random import sample

parser = argparse.ArgumentParser(description='')

parser.add_argument('--img_in', type=str, help='Dirección de la carpeta con las imágenes.')
parser.add_argument('--mask_in', type=str, help='Dirección de la carpeta con las máscaras.')
parser.add_argument('--img_out', type=str, help='Dirección de la carpeta con las imágenes.')
parser.add_argument('--mask_out', type=str, help='Dirección de la carpeta con las máscaras.')
parser.add_argument('--amount', type=int, help='Dirección de la carpeta con las imágenes.')
parser.add_argument('--split_percent', type=float, help='', default=0.33)



args = parser.parse_args()

# get file names
nema_names = next(walk(args.img_in), (None, None, []))[2]  # [] if no file
mask_names = next(walk(args.mask_in), (None, None, []))[2]  # [] if no file


nema_names.sort()
mask_names.sort()


if args.amount != None:
    indices = sample( range(len(nema_names)), args.amount )
    
    nema_names = list(map(nema_names.__getitem__, indices))
    mask_names = list(map(mask_names.__getitem__, indices))


X_train, X_val, y_train, y_val = train_test_split(nema_names, mask_names, test_size=args.split_percent, random_state=42)

def move_x_train(img):
    shutil.copy(args.img_in + '/' + img, args.img_out + '/train/' + img  )

def move_x_val(img):
    shutil.copy(args.img_in + '/' + img, args.img_out + '/val/' + img )

def move_y_train(img):
    
    #shutil.copy(args.mask_in + '/' + img, args.mask_out + '/train/' + img  )
    
    im = Image.open(args.mask_in + '/' + img)
    im = Image.fromarray(np.asarray(im.convert('P')), mode='P')
    im.save(args.mask_out + '/train/' + img)
    os.remove(args.mask_in + '/' + img)


def move_y_val(img):
    #shutil.copy(args.mask_in + '/' + img, args.mask_out + '/val/' + img )
    
    im = Image.open(args.mask_in + '/' + img)
    im = Image.fromarray(np.asarray(im.convert('P')), mode='P')
    im.save(args.mask_out + '/val/' + img)
    os.remove(args.mask_in + '/' + img)


# Crea output dir si no existe
if not os.path.exists(args.img_out + '/train'):
    os.makedirs(args.img_out + '/train')
    
# Crea output dir si no existe
if not os.path.exists(args.img_out + '/val'):
    os.makedirs(args.img_out + '/val')
    
# Crea output dir si no existe
if not os.path.exists(args.mask_out + '/train'):
    os.makedirs(args.mask_out + '/train')
    
# Crea output dir si no existe
if not os.path.exists(args.mask_out + '/val'):
    os.makedirs(args.mask_out + '/val')

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(move_x_train, X_train) 

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(move_x_val, X_val) 

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(move_y_train, y_train) 
    
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(move_y_val, y_val)     


















