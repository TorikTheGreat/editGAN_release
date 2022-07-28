#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
from PIL import Image, ImageDraw
import numpy as np


parser = argparse.ArgumentParser(description='Generador de máscaras a partir de anotaciones VIA.')

parser.add_argument('--img_in', type=str, help='Dirección de la carpeta con las imágenes.')
parser.add_argument('--label_file', type=str, help='Dirección del file JSON con las anotaciones.')
parser.add_argument('--mask_out', type=str, help='Dirección de la carpeta donde guardar las máscaras.')
parser.add_argument('--mask_edit_out', type=str, help='Dirección de la carpeta donde guardar las máscaras editadas.')
parser.add_argument('--img_edit_out', type=str, help='Dirección de la carpeta donde guardar las imágenes editadas.')

args = parser.parse_args()

def crop_img(img_name, new_name, img_dims, x_list):
    
    width, height = img_dims
    
    # load mask and image
    mask_path = args.mask_out + '/' + new_name
    img_path = args.img_in + '/' + img_name
    
    mask = Image.open ( mask_path )
    img = Image.open( img_path )
    
    # crop
    if min( x_list ) <= 128 :
        
        mask_edit = mask.crop(( 0, 0, 768, height  ))
        img_edit = img.crop(( 0, 0, 768, height  ))
        
        
    elif max( x_list ) >= 896:
        
        mask_edit = mask.crop(( 256, 0, width, height  ))
        img_edit = img.crop(( 256, 0, width, height  ))
        
    else:
        mask_edit = mask.crop(( 128, 0, 896, height  ))
        img_edit = img.crop(( 128, 0, 896, height  ))
    
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

def get_img_dims(img_name):
    path = args.img_in + '/' + img_name
    im = Image.open( path )
    return im.size


with open(args.label_file) as f:
    annotations = json.load(f)

img_data = annotations['_via_img_metadata']
img_keys = img_data.keys()

for i in img_keys:
   
    current_img = img_data[i]
    img_name = current_img['filename']
    img_dims = get_img_dims(img_name)
    
    
    # make a list of touples
    if current_img['regions']:
        x_list = current_img['regions'][0]['shape_attributes']['all_points_x']
        y_list = current_img['regions'][0]['shape_attributes']['all_points_y']
        coords = list(zip(x_list, y_list))
        new_img = Image.new("RGB", img_dims)
        img1 = ImageDraw.Draw(new_img)
        img1.polygon(coords, fill = 'white', outline = 'white')
        
        # save mask
        new_name = img_name[:-4] + '_mask.png'
        new_img = new_img.save(args.mask_out + '/' + new_name )
        print(f'Saved {img_name}\'s mask as {new_name} ')
        
        
        if args.mask_edit_out is not None:
            crop_img(img_name, new_name, img_dims, x_list)


    else:
        print(f'No annotation data for {img_name}. Skipping...')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    