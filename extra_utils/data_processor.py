#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from PIL import Image
from os import walk
import os
from random import sample
import shutil


parser = argparse.ArgumentParser(description='Cambia los nombres de las imagenes sinteticas al formato image_#.png y  \
                                 los de las máscaras al formato image_mask#.npy. Si recibe el argumento sample_size   \
                                 tomará una muestra aleatoria de ese tamaño de las imágenes y máscaras disponibles.')

parser.add_argument('img_in', type=str, help='Dirección de la carpeta con las imágenes.')
parser.add_argument('mask_in', type=str, help='Dirección de la carpeta con las máscaras.')
parser.add_argument('img_out', type=str, help='Carpeta donde copiar las imágenes' )
parser.add_argument('--sample_size', type=int, help='Cantidad de imágenes que copiar'  )



args = parser.parse_args()


# get file names
filenames = next(walk(args.img_in), (None, None, []))[2]  # [] if no file
filenames.sort()

masknames = next(walk(args.mask_in), (None, None, []))[2]  # [] if no file
masknames.sort()

if args.sample_size is not None :
    if args.sample_size <= len(filenames):
        
        #Elige al azar las imágenes y máscaras correspondientes
        filenames = sample(filenames,args.sample_size)
        masknames = [ mask for mask in masknames if mask[:-14]+mask[-9:] in filenames]
        filenames.sort()
        masknames.sort()
        
        
    else:
        print('sample_size demasiado grande para conjunto de datos.')


counter = 0

for img, msk in zip(filenames, masknames):
    
    # Renombra y copia imagen
    name = 'image_' + str(counter) + '.png'
    shutil.copy(args.img_in + '/' + img, args.img_out + '/' + name )
    print(f'Renamed {img} to {name}')
    
    # Reformatea, renombra y guarda máscara
    path = args.mask_in + '/' + msk
    mask = Image.open(path)
    img_npy = np.asarray(mask)[:,:,0]/255
    new_name ='image_mask' + str(counter) + '.npy' 

    np.save(args.img_out + '/' + new_name , img_npy)
    print(f'Mask {msk} saved as {new_name} \n')
    
    
    counter += 1





        
