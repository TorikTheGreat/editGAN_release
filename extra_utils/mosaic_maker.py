#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generador de mosáicos imagen-máscara

Este es un simple script para mostrar los resultados finales del conjunto
imagen-máscara creado. Aproxima la cantidad de imágenes pedida al cuadrado 
perfecto más cercano, poniendo las máscaras al lado de sus imágenes correspondientes
generando así un rectángulo de ejemplos elegidos al azar.
"""


import argparse
from os import walk
from PIL import Image
import random
from random import sample
random.seed(6)
import os
from math import sqrt, floor


def load_imgs(img_in, mask_in, img_num):
    """
    Carga un muestreo de las imágenes y sus máscaras correspondientes.

    Parámetros
    ----------
    img_in : str
        Dirección donde se encuentran las imágenes.
    mask_in : str
        Dirección donde se encuentran las máscaras.
    img_num : int
        Número de parejas imagen-máscara que cargar.

    Retorna
    -------
    imgs : list
        Lista con las imágenes muestreadas.
    masks : list
        Lista con las máscaras muestreadas

    """
    
    # Muestrea ejemplos
    img_names = next(walk(img_in), (None, None, []))[2]  # [] if no file
    img_names.sort()

    indices = sample( range(len(img_names)), img_num )

    img_names = list(map(img_names.__getitem__, indices))

    mask_names = next(walk(mask_in), (None, None, []))[2]  # [] if no file
    mask_names.sort()
    mask_names = list(map(mask_names.__getitem__, indices))
    
    # Devuelve las imágenes y máscaras
    img_names = [ img_in + '/' + img for img in img_names ]
    img_names.sort()
    mask_names = [ mask_in + '/' +  mask for mask in mask_names ]
    mask_names.sort()
    imgs = [ Image.open(img) for img in img_names  ]
    masks = [ Image.open(mask) for mask in mask_names ]
    return imgs, masks

#Código para obtener cuadrado perfecto más cercano tomado de https://www.geeksforgeeks.org/closest-perfect-square-and-its-distance/
def get_size(ex_num, res):
    
    """
    Calcula las dimensiones del cuadrado perfecto más cercano a aquel construido
    con ex_num cantidad de sub-cuadrados cuyo lado mide res.
    
    Parámetros
    ----------
    ex_num: int
        Cantidad de sub-cuadrados
    
    res: int
        Tamaño de los sub-cuadrados
        
    Retorna
    -------
    Dimensiones del cuadrado perfecto.
    
    """
    
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
    
def create_mosaic(img_in, mask_in, img_num, res=256):
    """
    Crea un mosáico con un muestreo aleatorio de parejas imagen-máscara

    Parameters
    ----------
    img_in : str
        Dirección donde se encuentran las imágenes.
    mask_in : str
        Dirección donde se encuentran las máscaras.
    img_num : int
        Cantidad de parejas imagen-máscara que incluir en el mosáico.
    img_out : str
        Dirección donde guardar el mosáico.
    res : int, opcional
        La resolución de cada imágen y máscara. El valor default es 256.

    Returns
    -------
    dst : pillow image
        El mosáico terminado.

    """
    
    imgs, masks = load_imgs(img_in, mask_in, img_num)
    
    dims = get_size(img_num/2, res)
    
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
    dst.show()
        
    return dst        




def main(args):
    
    # Crea output dir si no existe
    if not os.path.exists(args.img_out):
        os.makedirs(args.img_out)
    
    create_mosaic(args.img_in, args.mask_in, args.img_num).save(args.img_out + '/' + 'mosaic.png')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generador de mosáicos para las parejas imagen-máscara.')

    parser.add_argument('--img_in', type=str, help='Dirección de la carpeta con las imágenes.')
    parser.add_argument('--mask_in', type=str, help='Dirección de la carpeta con las máscaras.')
    parser.add_argument('--img_num', type=int, help='Número de imágenes que mostrar.')
    parser.add_argument('--img_out', type=str, help='Directorio donde poner el mosáico.')


    args = parser.parse_args()
    main(args)

 
 

    





