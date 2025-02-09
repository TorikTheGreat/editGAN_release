#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esta es una pequeña herramienta para producir máscaras a partir de un file .json
de anotaciones en el formato VIA. Sus enrtadas son img_in, la carpeta donde se
encuentran las imágenes que corresponden a las máscaras, y label_file, el documento
con las anotaciones en formato VIA.

Sus salidas son las siguientes:
    
    mask_out: directorio donde guardar las máscaras generadas sin modificación alguna
    mask_edit_out: directorio donde guardar las máscaras después de haber sido 
        recortadas y escaladas para que los nematodos queden en el centro.
    img_edit_out: directorio donde guardar las imágenes editadas de la misma manera
        que las máscaras para que mantengan su correspondencia.

"""

import argparse
import json
from PIL import Image, ImageDraw
import numpy as np
from os import walk
import os



parser = argparse.ArgumentParser(description='Generador de máscaras a partir de anotaciones VIA.')

parser.add_argument('--img_in', type=str, help='Dirección de la carpeta con las imágenes.')
parser.add_argument('--label_file', type=str, help='Dirección del file JSON con las anotaciones.')
parser.add_argument('--mask_out', type=str, help='Dirección de la carpeta donde guardar las máscaras.')
parser.add_argument('--mask_edit_out', type=str, help='Dirección de la carpeta donde guardar las máscaras editadas.')
parser.add_argument('--img_edit_out', type=str, help='Dirección de la carpeta donde guardar las imágenes editadas.')

args = parser.parse_args()

# Margen de pixeles al rededor de un nematodo recortado
crop_margin = 7

def crop_img(img, mask, new_name, img_dims, x_list, y_list):
    
    width, height = img_dims
    # Si el nematodo cabe en una caja 256x256:
    if  max(x_list) - min(x_list) <= 256 and max(y_list) - min(y_list) <= 256:
        
        # Determinar bordes
        left_border = max(0, round( ( min(x_list) + max(x_list) -256 )/2 ) )
        right_border = left_border + 256
        
        top_border = max(0, round( ( min(y_list) + max(y_list) -256 )/2 ) )
        bottom_border = top_border + 256
        
        # Ajusta si te saliste a la derecha o abajo
        if right_border > width:
            left_border -= right_border - width
            right_border -= right_border - width
        if bottom_border > height:
            top_border -= bottom_border - height
            bottom_border -= bottom_border - height
            
        # Crop
        mask_edit = mask.crop(( left_border, top_border, right_border, bottom_border  ))
        img_edit = img.crop(( left_border, top_border, right_border, bottom_border  ))
        
    # Si no, hay que cropear al rededor del nematodo y escalar. 
    # NOTA: es importante recortar un cuadrado al rededor del nematodo para que a la hora 
    #       de escalar se mantengan las proporciones
    else:
	# Establece la dimensión más grande de la caja que contiene al nematodo
        square_side = max( max(x_list) - min(x_list), max(y_list) - min(y_list) ) + 2*crop_margin
        
        # Determinar bordes
        if max(x_list) - min(x_list) > max(y_list) - min(y_list):
            left_border = min(x_list) - crop_margin
            right_border = left_border + square_side
        	
            top_border = max(0, round( ( min(y_list) + max(y_list) - square_side )/2 ) )
            bottom_border = top_border + square_side
        
        elif max(x_list) - min(x_list) < max(y_list) - min(y_list):
            top_border = min(y_list) - crop_margin
            bottom_border = top_border + square_side
        	
            left_border = max(0, round( ( min(x_list) + max(x_list) - square_side )/2 ) )
            right_border = left_border + square_side
            
        # Ajusta si te saliste a la derecha o abajo
        if right_border > width:
            left_border -= right_border - width
            right_border -= right_border - width
        if bottom_border > height:
            top_border -= bottom_border - height
            bottom_border -= bottom_border - height
	        
        # Crop
        mask_edit = mask.crop(( left_border, top_border, right_border, bottom_border  ))
        img_edit = img.crop(( left_border, top_border, right_border, bottom_border  ))
        
        # Resize images
        mask_edit = mask_edit.resize((256,256))
        img_edit = img_edit.resize((256,256))
    
    # keep mask monocrhomatic
    conversion = np.array(mask_edit)
    converted = np.where( conversion == 255,255,0)
    mask_edit = Image.fromarray(converted.astype('uint8'))
    
    # save imgs
    
    mask_edit = mask_edit.save(args.mask_edit_out + '/' + new_name[:-4] + '_edit.png' )
    img_edit = img_edit.save(args.img_edit_out + '/' + img_name[:-4] + '_edit.png' )


with open(args.label_file) as f:
    annotations = json.load(f)

# get image file names
filenames = next(walk(args.img_in), (None, None, []))[2]  # [] if no file
filenames.sort()

img_data = annotations['_via_img_metadata']
img_keys = img_data.keys()

for i in img_keys:
    
    
    
    current_img = img_data[i]
    img_name = current_img['filename']
    
    if img_name in filenames:
    
        # Carga imagen
        path = args.img_in + '/' + img_name
        img = Image.open( path )
        
        img_dims = img.size
        
        # Crea nombre para la máscara
        new_name = img_name[:-4] + '_mask.png'
        
        # make a list of touples
        if current_img['regions']:
            x_list = current_img['regions'][0]['shape_attributes']['all_points_x']
            y_list = current_img['regions'][0]['shape_attributes']['all_points_y']
            coords = list(zip(x_list, y_list))
            new_img = Image.new("RGB", img_dims)
            img1 = ImageDraw.Draw(new_img)
            img1.polygon(coords, fill = 'white', outline = 'white')
            
            # save mask
            new_img = new_img.save(args.mask_out + '/' + new_name )
            print(f'Saved {img_name}\'s mask as {new_name} ')
            
            
            if args.mask_edit_out is not None:
                # Carga máscara
                mask_path = args.mask_out + '/' + new_name
                mask = Image.open ( mask_path )
                crop_img(img, mask, new_name, img_dims, x_list, y_list)
                mask.close()
    
    
        else:
            print(f'No label data for {img_name}. Skipping...')
            
        # Cierra las imágenes
        img.close()
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
