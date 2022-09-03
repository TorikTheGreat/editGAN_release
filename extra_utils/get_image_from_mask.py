#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 18:21:52 2022

@author: marco
"""


#from flask import Flask, jsonify, request
#from flask_cors import CORS, cross_origin
import torch
#import flask
import imageio
#torch.manual_seed(0)
import json
import pickle
#from flask import Blueprint, render_template
import os
import cv2

device_ids = [0]
from PIL import Image
import timeit
from utils.poisson_image_editing import poisson_edit
from utils.data_utils import *
#from utils.model_utils import *
import numpy as np
import argparse
import copy
from io import BytesIO
from os import walk
#from models.EditGAN.EditGAN_tool import Tool

#delete this when more competent.
try:
    from utils.model_utils import *
    print('Funciono a la primera')
except:
    print('strike 1')
try:
    from utils.model_utils import *
    print('Funciono a la seguna')
except:
    print('strike 2')
try:
    from utils.model_utils import *
    print('Tercera es la vencida?')
except:
    print(':(')

from models.EditGAN.EditGAN_tool import Tool

test_path = '/home/marco/Desktop/proyecto_graduacion/editGAN_release/sol_one/mask_in'

parser = argparse.ArgumentParser(description='Generador de máscaras en el formato que ocupa editgan. Color blanco: nematodo, color negro: fondo')

parser.add_argument('mode', type=int, help='1: Run a sampling from Stylegan2\'s latent space. \n2: Generate masks from real nematodes.')

parser.add_argument('--mask_in_path', type=str, help='Directorio de las máscaras de entrada.')

parser.add_argument('--img_out_path', type=str, help='Directorio de las imágenes de salida.')

parser.add_argument('--mask_out_path', type=str, help='Directorio de las imágenes de salida.')

parser.add_argument('--sampling_amount', type=int, help='Amount of image-mask pairs to generate.')


args = parser.parse_args()



tool = Tool()



if args.mode == 1:
    for i in range(args.sampling_amount):
        img_out, img_seg_final, latent = tool.run_sampling()
        imageio.imsave(args.img_out_path + '/' + 'image_' + str(i) + '.png', img_out)
        imageio.imsave(args.mask_out_path + '/' +'mask_' + str(i) +'.png',img_seg_final)

    
# elif args.mode == 2:
#     # get file names
#     filenames = next(walk(args.mask_in_path), (None, None, []))[2]  # [] if no file
#     print(filenames)
    
#     for i in filenames:
#         print(i)
#         path = args.mask_in_path + '/' + i
#         img = Image.open(path)
#         #img_npy = np.asarray(img)
        
#         #print(f'IM.MODE: {img.mode}')
#         if img.mode == 'RGBA':
#             img = img.convert('RGB')
            
            
#         #img = img[0,:]
        
#         img_out, img_seg_final, optimized_latent, optimized_noise = tool.run_embedding(img)
        
#         img_out = Image.fromarray(img_out)
        
#         imageio.imsave(args.img_out_path + '/' + i,img_out)
#         #imageio.imsave(args.img_out_path + '/' +'img_seg_final_' + i[0:-4] +'.png',img_seg_final)

    
elif args.mode == 2:
    # get file names
    filenames = next(walk(args.mask_in_path), (None, None, []))[2]  # [] if no file
    print(filenames)
    
    for i in filenames:
        print(i)
        path = args.mask_in_path + '/' + i
        img = Image.open(path)
        #img_npy = np.asarray(img)
        
        #print(f'IM.MODE: {img.mode}')
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img_out, img_seg_final, optimized_latent, optimized_noise = tool.run_embedding(img)
        
        #imageio.imsave(args.img_out_path + '/' + i,img_out)
        imageio.imsave(args.img_out_path + '/' +'img_seg_final_' + i[0:-4] +'.png',img_seg_final)
        
        
elif args.mode == 3:
    # get file names
    filenames = next(walk(args.mask_in_path), (None, None, []))[2]  # [] if no file
    print(filenames)
    
    for i in filenames:
        print(i)
        path = args.mask_in_path + '/' + i
        img = Image.open(path)

        
        #print(f'IM.MODE: {img.mode}')
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img_out, img_seg_final, optimized_latent, optimized_noise = tool.run_embedding(img)

        img_out = np.squeeze(img_out, axis=0)


        
        imageio.imsave(args.img_out_path + '/' + 'image_' + str(i) + '.png', img_out)
        imageio.imsave(args.img_out_path + '/' +'img_seg_final_' + i[0:-4] +'.png',img_seg_final)
        
        
        
        
        
        
        
        
        
        
        
