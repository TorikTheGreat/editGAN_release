#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para entrenar un support vector machine como filtro para las máscaras 
producidas por EditGAN.
"""

# Código inspirado en https://medium.com/analytics-vidhya/image-classification-using-machine-learning-support-vector-machine-svm-dc7a0ec92e01

import pandas as pd
import numpy as np
from os import walk
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import cv2
from sklearn.preprocessing import MinMaxScaler
from scipy import fftpack
import argparse
from tqdm import tqdm
import os



def load_svm_class(class_path, class_index, data_array, target_array):
    
    """
    Función para cargar una clase del svm. Recibe y devuelve un data array y 
    un target array; la idea de hacerlo de esta manera es que se pueda llamar
    de forma consecutiva, una vez por clase formando así una cadena y manteniendo
    el orden en que se llamó en los array finales.
    
    Parámetros
    ----------
    class_path: str
        La dirección donde se encuentran las imágenes de la clase que cargar.
    
    class_index: int
        Las clases se codifican con un index numérico, y esta variable corresponde
        al index de la clase a cargar.
    
    data_array: array
        El estado del data_array antes de cargar la clase.
    
    target_array: array
        El estado del target_array antes de cargar la clase.
        
    Retorna
    -------
    data_array: array
        Un array donde cada fila corresponde a uno de los ejemplos cargados y 
        las columnas corresponden a sus 7 momentos de Hu, su perímetro y la 
        magnitud de las 5 componentes de mayor frecuencia de su transformada de
        fourier, en ese orden.
    
    target_array: list
        Es básicamente una lista donde cada elemento corresponde a una fila de 
        data_array y contiene el index de la clase del ejemplo. 
    """

    img_names = next(walk(class_path), (None, None, []))[2]  # [] if no file
    
    for a in tqdm(range(len(img_names))):
        i = img_names[a]
        path = class_path + '/' + i
        
        current_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        _,current_img = cv2.threshold(current_img, 128, 255, cv2.THRESH_BINARY)
        
        contours,hierarchy = cv2.findContours(current_img, 1, 2)
        perimeter = cv2.arcLength(contours[0],True)
    
        moments = cv2.moments(contours[0])
        hu_moments = cv2.HuMoments(moments)
        
        hu_moments = np.concatenate( (hu_moments, [[perimeter]]), axis = 0 )
        
        F = fftpack.fftn(current_img)
        F_magnitude = np.abs(F).tolist() [0]
        F_magnitude.sort(reverse=True)
        F_magnitude = np.array( [F_magnitude[: 5]] )
        hu_moments = np.concatenate( (hu_moments, F_magnitude.transpose()), axis = 0 )
        
        data_array.append(hu_moments.flatten())
    
        target_array.append(class_index)
        #print(f'Loaded img {i}')
    
    return data_array, target_array

def load_svm_data(good_path, bad_path, save_path):
    
    """
    Función para cargar todos los datos de todas las clases para las que entrenará
    el SVM. Actualmente está hecho para dos clases solamente, máscaras buenas y 
    máscaras malas, pero se podría fácilmente modificar para más clases (por 
    ejemplo para que clasifique diferentes especies de nematodo). 
    
    Para que el SVM funcione correctamente es necesario usar un MinMaxScaler
    en los datos, y es importante usar el mismo scaler a la hora de hacer 
    inferencia así que el scaler se guardará en save_path
    
    Parámetros
    ----------
    good_path: str
        La dirección donde se encuentran las imágenes de las máscaras buenas.
    
    bad_path: str
        La dirección donde se encuentran las imágenes de las máscaras malas.
    
    save_path: str
        La dirección donde se guardará el scaler.
    
    Retorna
    -------
    df: data frame
        Un dataframe que contiene todos los datos, un ejemplo por fila. Las 
        columnas corresponden a sus 7 momentos de Hu, su perímetro y la magnitud
        de las 5 componentes de mayor frecuencia de su transformada de fourier
        y el index de la clase a la que corresponde el ejemplo, en ese orden. 
    """
    
    data_array = []
    target_array = []
    
    print('\nLoading good masks...')
    
    data_array, target_array = load_svm_class(good_path, 0, data_array, target_array)
    
    print('\nLoading bad masks...')
    data_array, target_array = load_svm_class(bad_path, 1, data_array, target_array)
    
    print('\nScaling data...')
    
    
    squishy_data = np.array(data_array)
    squishy_targets = np.array(target_array)

    df = pd.DataFrame(squishy_data)

    scaler = MinMaxScaler()
    d = scaler.fit_transform(df)
    df = pd.DataFrame(d)

    # pon columna con targets
    df['target'] = squishy_targets
    
    scaler_path = save_path + '/scaler.sav'
    
    print('\nSaving scaler as {scaler_path}')
    
    pickle.dump(scaler, open(scaler_path, 'wb'))

    return df


def train_svm(good_path, bad_path, save_path, param_grid):
    
    """
    Entrena un SVM para determinar si una máscara es buena o mala.
    
    Parámetros
    ----------
    good_path: str
        La dirección donde se encuentran las imágenes de las máscaras buenas.
    
    bad_path: str
        La dirección donde se encuentran las imágenes de las máscaras malas.
    
    save_path: str
        La dirección donde se guardará el scaler y el modelo elejido por el 
        grid search.
        
    param_grid: dict
        Diccionario que contiene los parámetros que incluir en el grid search
        Ejemplo:
        {'C':[1,4,5,6,10],'gamma':[50,85, 90, 100, 110, 115],'kernel':['sigmoid', 'rbf','poly']}
    
    """
    
    # Crea output dir si no existe
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    df = load_svm_data(good_path, bad_path, save_path)
    
    print('\nTraining SVM...')


    x=df.iloc[:,:-1] 
    y=df.iloc[:,-1] 
    
    svc=svm.SVC(probability=True)
    model=GridSearchCV(svc,param_grid, n_jobs=-1, verbose=4)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,stratify=y)
    
    model.fit(x_train,y_train)
    
    y_pred=model.predict(x_test)
    
    #Save the model
    svm_path = save_path + '/best_svm.sav'
    print(f'\nSaving model as {svm_path}')
    pickle.dump(model, open(svm_path, 'wb'))

    print(f"\nThe model is {accuracy_score(y_pred,y_test)*100}% accurate")
    
    print(f'\nThe best model params: {model.best_params_}')
    
    
    
def main(args):
    param_grid={'C':[1,4,5,6,10],'gamma':[50,85, 90, 100, 110, 115],'kernel':['sigmoid', 'rbf','poly']}
    train_svm(args.good_path, args.bad_path, args.save_path, param_grid)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Entrenador para un SVM que filtra las máscaras de nematodos según su calidad.')

    parser.add_argument('--good_path', type=str, help='Dirección de la carpeta con los ejemplos de máscaras buenas')
    parser.add_argument('--bad_path', type=str, help='Dirección de la carpeta con los ejemplos de máscaras malas')
    parser.add_argument('--save_path', type=str, help='Dirección donde guardar el checkpoint del svm y el scaler.')

    args = parser.parse_args()
    main(args)
    





