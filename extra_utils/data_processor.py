#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Organizador de ejemplos de entrenamiento etiquetados para datasetGAN

Para entrenar a datasetGAN es necesario proveer un pequeño conjunto de imágenes 
etiquetadas para que el modelo pueda aprender a detectar las representaciones 
del espacio latente de StyleGAN que se convertirán en las máscaras. Las imágenes
deben estar nombradas en el formato image_#.png y las máscaras deben ser convertidas
a un array de numpy y guardadas como image_mask#.npy. Todos los files deben
guardarse en una misma carpeta.

Este script selecciona al azar un número de imágenes determinado por el parámetro
sample_size. Si este parámetro no es recibido, entonces procesa todas las parejas
imagen-máscara que encuentra.

"""


import argparse
import numpy as np
from PIL import Image
from os import walk
import os
from random import sample
import shutil


def process_data(img_in, mask_in, img_out, sample_size=None):
    
    """Selecciona al azar un subconjunto de entrenamiento para datasetGAN
    
    Mueve las imágenes y cambia su nombre al formato image_#.png, transforma
    las máscaras a un array y cambia su nombre al formato image_mask#.npy.
    
    Si el argumento sample_size no es recibido, procesará el conjunto entero.
    
    Parámetros
    ----------
    img_in: str
        Dirección de la carpeta donde se encuentran las imágenes de nematodos
    
    mask_in: str
        Dirección de la carpeta donde se encuentran las máscaras.
        
    img_out: str
        Dirección donde guardar las imágenes y máscaras procesadas.
    
    sample_size: int, opcional
        Cantidad de pares imagen-máscara que procesar.
    """
    
    # Crea output dir si no existe
    if not os.path.exists(img_out):
        os.makedirs(img_out)
    
    # get file names
    filenames = next(walk(img_in), (None, None, []))[2]  # [] if no file
    filenames.sort()
    
    masknames = next(walk(mask_in), (None, None, []))[2]  # [] if no file
    masknames.sort()
    
    if sample_size is not None :
        if sample_size <= len(filenames):
            
            #Elige al azar las imágenes y máscaras correspondientes
            filenames = sample(filenames,sample_size)
            masknames = [ mask for mask in masknames if mask[:-14]+mask[-9:] in filenames]
            filenames.sort()
            masknames.sort()
            
            
        else:
            print('sample_size demasiado grande para conjunto de datos.')
    
    
    counter = 0
    
    for img, msk in zip(filenames, masknames):
        
        # Renombra y copia imagen
        name = 'image_' + str(counter) + '.png'
        shutil.copy(img_in + '/' + img, img_out + '/' + name )
        print(f'Renamed {img} to {name}')
        
        # Reformatea, renombra y guarda máscara
        path = mask_in + '/' + msk
        mask = Image.open(path)
        img_npy = np.asarray(mask)[:,:,0]/255
        new_name ='image_mask' + str(counter) + '.npy' 
    
        np.save(img_out + '/' + new_name , img_npy)
        print(f'Mask {msk} saved as {new_name} \n')
        
        
        counter += 1

def main(args):
    process_data(args.img_in, args.mask_in, args.img_out, args.sample_size)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Cambia los nombres de las imagenes sinteticas al formato image_#.png y  \
                                     los de las máscaras al formato image_mask#.npy. Si recibe el argumento sample_size   \
                                     tomará una muestra aleatoria de ese tamaño de las imágenes y máscaras disponibles.')

    parser.add_argument('img_in', type=str, help='Dirección de la carpeta con las imágenes.')
    parser.add_argument('mask_in', type=str, help='Dirección de la carpeta con las máscaras.')
    parser.add_argument('img_out', type=str, help='Carpeta donde copiar las imágenes' )
    parser.add_argument('--sample_size', type=int, help='Cantidad de imágenes que copiar'  )



    args = parser.parse_args()
    main(args)
