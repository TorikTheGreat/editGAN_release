#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtro para eliminar las máscaras de peor calidad. Consta de dos etapas:
1- Un filtro que toma todos los contornos presentes en la máscara y descarta todos excepto el más grande.
   Si el área del más grande no supera el umbral min_area, la máscara y su imagen se desechan.
2- Un support vector machine, entrenado por aparte, cuyos parámetros son los 7 momentos de Hu, el perímetro
   del contorno y la magnitud de las 5 componentes de mayor frecuencia de la transformada de fourier de la 
   máscara.

"""

import time
import concurrent.futures
import numpy as np
import cv2
import argparse
from os import walk
import shutil
import pickle
from scipy import fftpack



def process_image(params):
    
    """
    Filtra un par imagen-máscara juzgando la calidad de la máscara.
    
    Su única entrada es un diccionario con la siguiente lista de parámetros, en el orden presentado. 
    Fue hecho de esta manera para que fuera más fácil paralelizar el filtrado, agilizando el 
    procesamiento de conjuntos grandes.
    
    Parámetros
    ----------
    img_in: str
        Dirección donde se encuentran las imágenes por filtrar.
    
    mask_in: str
        Dirección donde se encuentran las máscaras por filtrar.
        
    img_out: str
        Dirección donde guardar las imágenes que son aceptadas por el filtro.
        
    mask_out: str
        Dirección donde guardar las máscaras que son aceptadas por el filtro.
        
    bad_out: str
        Dirección donde guardar las máscaras que son rechazadas por el filtro.
        Si es None, no guarda estas máscaras.
    
    min_area: int
        Umbral de área ocupada por el namatodo. Máscaras con un area inferior 
        serán rechazadas.
    
    model: svm
        Modelo Support vector machine para filtrar las máscaras. Debe haber sido
        entrenado con probability=True.
        
    scaler: MinMaxScaler
        Scaler utilizado para entrenar el SVM. Si se usa uno diferente podría
        llevar a resultados inesperados.
        
    svm_proba: float
        Umbral de probabilidad para que el SVM acepte una máscara.
        
    mask_name: str
        Nombre del file que contiene la máscara a procesar.
    
    """
    
    model = params['model']
    scaler = params['scaler']
    mask_name = params['mask_name']    
    
    
    print(f'Procesando {mask_name}')
    
    image = cv2.imread( params['mask_in'] + '/' + mask_name , 0)
    new_image = np.zeros( ( image.shape[0],image.shape[1] ,3), np.uint8 )

    #get contours
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # Remove all contours below specified area
    contours = [ cnt for cnt in contours if cv2.contourArea(cnt) > params['min_area'] ]
    
    if len(contours) != 0:
        #get biggest contour
        
        c = max(contours, key = cv2.contourArea)
        im = cv2.fillPoly(new_image, pts =[c], color=(255,255,255))
        moments = cv2.moments(c)
        
        hu_moments = cv2.HuMoments(moments)
        
        perimeter = cv2.arcLength(c,True)
        l = np.concatenate( (hu_moments, [[perimeter]]), axis = 0 ).flatten().reshape(1,-1)
                
        current_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _ , current_img = cv2.threshold(current_img, 128, 255, cv2.THRESH_BINARY)
        F = fftpack.fftn(current_img)
        F_magnitude = np.abs(F).tolist()[0]
        F_magnitude.sort(reverse=True)
        F_magnitude = F_magnitude[: 5]
        
        l = np.array( np.concatenate( (l[0], F_magnitude), axis = 0 ) ).reshape(1, -1)

        l = scaler.transform(l)
                
        probability=model.predict_proba(l)
        
        if probability[0][0] >= params['svm_proba']:
            
            cv2.imwrite(params['mask_out'] + '/' + mask_name , im)
                        
            image_name = mask_name.replace('mask', 'image')
            shutil.copyfile(  params['img_in'] + '/' + image_name ,  params['img_out'] + '/' + image_name )
            
        else:
            if params['bad_out'] != None:
                cv2.imwrite(params['bad_out'] + '/' + mask_name , im)
            


def main(args):
    
    masknames = next(walk(args.mask_in), (None, None, []))[2]  # [] if no file
    masknames.sort()
    
    
    model = pickle.load(open(args.svm_path, 'rb'))
    scaler = pickle.load(open(args.scaler_path, 'rb'))
    
    param = { 
        'img_in': args.img_in,
        'mask_in': args.mask_in,
        'img_out': args.img_out,
        'mask_out': args.mask_out,
        'bad_out': args.bad_out,
        'min_area': args.min_area,
        'model': model,
        'scaler': scaler,
        'svm_proba': args.svm_proba,
        'mask_name': ''
        }
    
    params = []
    
    for i in masknames:
        temp = param.copy()
        temp['mask_name'] = i
        params.append(temp)
    
    
    t1 = time.perf_counter()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_image, params) 
    
    
    t2 = time.perf_counter()
    time_dif = (t2-t1)/(60*60)
    
    print(f'Finished in {time_dif} hours')
    

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Filtro para exluir las imágenes de menor calidad producidas por EditGAN')
    
    parser.add_argument('--img_in', type=str, help='Dirección de la carpeta con las imágenes.')
    parser.add_argument('--mask_in', type=str, help='Dirección de la carpeta con las máscaras.')
    parser.add_argument('--img_out', type=str, help='Carpeta donde copiar las imágenes.')
    parser.add_argument('--mask_out', type=str, help='Carpeta donde poner las mascaras filtradas' )
    parser.add_argument('--bad_out', type=str, help='Carpeta donde poner las mascaras rechazadas por el filtro' )
    parser.add_argument('--min_area', type=int, help='Mínima cantidad de área que filtrar', default=2600  )
    parser.add_argument('--svm_path', type=str, help='La dirección del checkpoint del svm' )
    parser.add_argument('--scaler_path', type=str, help='La dirección del scaler usado para entrenar el svm' )
    parser.add_argument('--svm_proba', type=float, help='Probabilidad mínima para la desición del svm.', default=0.80)
    
    args = parser.parse_args()
    
    main(args)










