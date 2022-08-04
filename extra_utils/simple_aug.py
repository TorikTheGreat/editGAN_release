#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os import walk
import os
from PIL import Image, ImageOps
import argparse

parser = argparse.ArgumentParser(description='Aumentador básico de imágenes. Actualmente rota y refleja.')

parser.add_argument('img_in', type=str, help='Dirección de la carpeta con las imágenes.')

args = parser.parse_args()

# get file names
filenames = next(walk(args.img_in), (None, None, []))[2]  # [] if no file
filenames.sort()

for i in filenames:
    
    print(f'Augmenting {i}...')
    
    name = i[:-4]
    
    # abrir file
    p0 = Image.open(args.img_in + '/' + i)
    
    # Por cada imagen puedo obtener 8 diferentes permutaciones con rotaciones y mirrors
    p0f = ImageOps.flip(p0)
    p0m = ImageOps.mirror(p0)
    p90 = p0.rotate(90)
    p90f = ImageOps.flip(p90)
    p90m = ImageOps.mirror(p90)
    p180 = p0.rotate(180)
    p270 = p0.rotate(270)
    
    # Guardar todas
    p0f.save(args.img_in + '/' + name +'_flip.png')
    p0m.save(args.img_in + '/' + name +'_mirror.png')
    p90.save(args.img_in + '/' + name +'_90.png')
    p90f.save(args.img_in + '/' + name +'_90_flip.png')
    p90m.save(args.img_in + '/' + name +'_90_mirror.png')
    p180.save(args.img_in + '/' + name +'_180.png')
    p270.save(args.img_in + '/' + name +'_270.png')
    
    p0.close()