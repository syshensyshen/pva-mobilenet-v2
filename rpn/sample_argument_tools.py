import os
import numpy as np
from numpy.matlib import repmat
import random
import sys
import math
import cv2

def equalizeHist3D(img):
    for item in range(0, img.shape[2]):
        img[:,:,item] = cv2.equalizeHist(img[:,:,item]) 
        
    return img

def normalization3D(img):
    for item in range(0, img.shape[2]):
        img[:,:,item] = (img[:,:,item] - np.min(img[:,:,item]) ) / (np.max(img[:,:,item]) - np.min(img[:,:,item])) * 255
    
    return img
        
def rotation3D(img, angle):
    h,w = img.shape[:2]
    center_x = (w >> 1)
    center_y = (h >> 1)
    sin_angle = np.fabs(np.sin(np.radians(angle)))
    cos_angle = np.fabs(np.cos(np.radians(angle)))
    new_h =int( w * sin_angle + h * cos_angle )
    new_w =int( h * sin_angle + w * cos_angle )    
    
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    rotated = cv2.warpAffine(img, M, (new_w, new_h))
    
    return rotated

def generatorNoise3D(img, noise_num):   
    
    h,w,c = img.shape
    
    size = h * w
    
    inds_index = np.random.random_integers(0, size - 1, noise_num)
    
    inds_value = np.random.random_integers(0, 255, noise_num)

    inds_value = inds_value[:,np.newaxis]
    
    inds_value = np.repeat(inds_value, 3, axis=1)    
    
    img = img.reshape((size, c))
    
    img[inds_index, :] = img[inds_index, :] + inds_value.reshape((noise_num, 3))
    
    img = img.reshape((h,w,c))
    
    np.minimum(img, 255)
    
    return img

def generatorNoise(img, noise_num):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    v = hsv[:, :, 2]
    
    def addNoise(img, noise_num):       
    
        h, w = img.shape[:2]
        size = h * w
        
        inds_index = np.random.random_integers(0, size - 1, noise_num)
        
        inds_value = np.random.random_integers(0, 255, noise_num)   
        
        img = img.reshape((size, 1))
        
        img[inds_index, :] = img[inds_index, :] + inds_value.reshape((noise_num, 1))
        
        img = img.reshape((h,w))
        
        np.minimum(img, 255)
        
        return img
    
    hsv[:, :, 2] = addNoise(v, noise_num)
    
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 
    
    return img


def generatorBlur(img, kernel, sigma):
    
    #h,w,c = img.shape
    
    img = cv2.GaussianBlur(img, (kernel, kernel), sigma)
    
    return img   

def gaussianLightimngArgument(img, scale, power, center_x, center_y):    
    height, width = img.shape[:2]

    R = np.sqrt(center_x**2 + center_y**2) * scale
    if 3 == len(img.shape):
        r_idx_array, c_idx_array = np.where(img[:, :, 0] < 256)
        mask_y = r_idx_array.reshape(height, width)
        mask_x = c_idx_array.reshape(height, width) 
    else:
        r_idx_array, c_idx_array = np.where(img[:, :] < 256)
        mask_y = r_idx_array.reshape(height, width)
        mask_x = c_idx_array.reshape(height, width)         

    #mask_x = repmat(center_x, height, width)
    #mask_y = repmat(center_y, height, width)

    x1 = np.arange(width)
    x_map = repmat(x1, height, 1)

    y1 = np.arange(height)
    y_map = repmat(y1, width, 1)
    y_map = np.transpose(y_map)

    Gauss_map = np.sqrt((x_map-mask_x)**2+(y_map-mask_y)**2) 

    Gauss_map = np.exp(-0.5*Gauss_map/R) * power
    
    if 3 == len(img.shape): 
        Gauss_map = Gauss_map[:,:,np.newaxis]
        Gauss_map = np.repeat(Gauss_map, 3, axis=2)      
    
    illumination_img = img * Gauss_map
    illumination_img = np.maximum(np.minimum(illumination_img, 255), 0)
    
    #if 3 == len(img.shape):
        #illumination_img = normalization3D(illumination_img)  
    #else:
        #illumination_img = (illumination_img - np.min(illumination_img) ) / (np.max(illumination_img) 
        #\- np.min(illumination_img)) * 255

    #if 3 == len(img.shape):
        #MAX = 255
        #inds = np.where(
            #(illumination_img[:, :, 0] > MAX) &
            #(illumination_img[:, :, 1] > MAX) &
            #(illumination_img[:, :, 2] > MAX) )[0]
        #illumination_img[inds, :] = 255
        
        #MIN = 0
        #inds = np.where(
            #(illumination_img[:, :, 0] < MIN) &
            #(illumination_img[:, :, 1] < MIN) &
            #(illumination_img[:, :, 2] < MIN) )[0]
        #illumination_img[inds, :] = 0
    #else:
        #MAX = 255
        #inds = np.where(
            #(illumination_img[:, :] > MAX) )[0]
        #illumination_img[inds, :] = 255
        
        #MIN = 0
        #inds = np.where(
            #(illumination_img[:, :] < MIN) )[0]
        #illumination_img[inds, :] = 0        

    illumination_img = np.uint8(illumination_img)
    
    return illumination_img    

def linerLightingArgument(img, power=1.2, scale=0.4, center=None):
    height, width = img.shape[:2]
    if center is None:
        center =  random.randint(0, width), random.randint(0, height)
        # center =1000,1000

    if 3 == len(img.shape):
        r_idx_array, c_idx_array = np.where(img[:, :, 0] < 256)
        r_idx_array = r_idx_array.reshape(height, width)
        c_idx_array = c_idx_array.reshape(height, width) 
    else:
        r_idx_array, c_idx_array = np.where(img[:, :] < 256)
        r_idx_array = r_idx_array.reshape(height, width)
        c_idx_array = c_idx_array.reshape(height, width) 

    radius = int(math.sqrt(height ** 2 + width ** 2))  #
    distance = np.sqrt((r_idx_array - center[1]) ** 2 + (c_idx_array - center[0]) ** 2)
    scale = max(0, scale)
    brightness = power * (radius - scale * distance) / radius
    
    if 3 == len(img.shape):
        brightness = brightness[:, :, np.newaxis]
        brightness = np.repeat(brightness, 3, axis=2)
        
    res_img = img * brightness
    res_img = np.maximum(np.minimum(res_img, 255), 0)
    res_img = np.uint8(res_img)

    return res_img

def illumination(img, scale, center_x, center_y, power, lightscale):
    num = random.randint(0, 2)
    #print num, scale, center_x, center_y, power, lightscale
    num = 0
    if 0 == num:
        return gaussianLightimngArgument(img, scale, power, center_x, center_y)
    elif 1 == num:
        return linerLightingArgument(img, power, lightscale, (center_x, center_y))
    else:
        gaussian = gaussianLightimngArgument(img, scale, power, center_x, center_y)
        center_x = np.random.randint(500, img.shape[1] - 500)
        center_y = np.random.randint(500, img.shape[0] - 500)        
        liner = linerLightingArgument(img, power, lightscale, (center_x, center_y))
        normalization = (gaussian / 2) + (liner / 2)
        return normalization
    
def Get_distance(img, center):
    height, width = img.shape[:2]
    if center is None:
        center = random.randint(0, width), random.randint(0, height)

    if 3 == len(img.shape):
        r_idx_array, c_idx_array = np.where(img[:, :, 0] < 256)
        r_idx_array = r_idx_array.reshape(height, width)
        c_idx_array = c_idx_array.reshape(height, width)
    else:
        r_idx_array, c_idx_array = np.where(img[:, :] < 256)
        r_idx_array = r_idx_array.reshape(height, width)
        c_idx_array = c_idx_array.reshape(height, width)

    distance = np.sqrt((r_idx_array - center[1]) ** 2 + (c_idx_array - center[0]) ** 2)
    
    return distance

def Get_Gauss_map(img, center_1, center_2, power, scale):
    distance = Get_distance(img, center_1)
    radius = int(math.sqrt(center_2[0] ** 2 + center_2[1] ** 2)) * scale  #
    Gauss_map = np.exp(-0.5 * distance / radius) * power
    if 3 == len(img.shape):
        Gauss_map = Gauss_map[:, :, np.newaxis]
        Gauss_map = np.repeat(Gauss_map, 3, axis=2)    
    return img * Gauss_map

def Get_Linear_map(img, center_1, center_2, power, scale):
    distance = Get_distance(img, center_1)
    radius = int(math.sqrt(center_2[0] ** 2 + center_2[1] ** 2))
    Linear_map = power * (radius - distance/(np.exp(scale)-1)) / radius
    if 3 == len(img.shape):
        Linear_map = Linear_map[:, :, np.newaxis]
        Linear_map = np.repeat(Linear_map, 3, axis=2)    
    return img * Linear_map

def lighting(img, center_1, center_2,  power, scale, light_type = None):
    
    if light_type in ["guass", 0]:
        res_map = Get_Gauss_map(center_1, center_2, power, scale)
    elif light_type in ["linear", 1]:
        res_map = Get_Linear_map(center_1, center_2, power, scale)
    else:
        print("light_type error!")
        return

    if 3 == len(img.shape):
        res_map = res_map[:, :, np.newaxis]
        res_map = np.repeat(res_map, 3, axis=2)

    res_img = img * res_map
    res_img = np.maximum(np.minimum(res_img, 255), 0)
    return res_img


def lightexchange(img):
    center_x = (img.shape[1] >> 1) + np.random.randint(-10, 10)
    center_y = (img.shape[0] >> 1) + np.random.randint(-10, 10)  
    center_1 = (center_x, center_y)
    scale = random.uniform(1, 5)
    power = random.uniform(0.7, 1.3)   
    center_x = (img.shape[1] >> 1) + np.random.randint(-10, 10)
    center_y = (img.shape[0] >> 1) + np.random.randint(-10, 10)  
    center_2 = (center_x, center_y)
    #illumination_img = Get_Linear_map(img, center_1, center_2, power, scale)
    select_num = np.random.randint(0, 1)
    if 1 == select_num:
       illumination_img = Get_Gauss_map(img, center_1, center_2, power, scale)
    elif 0 == select_num:
       illumination_img = Get_Linear_map(img, center_1, center_2, power, scale)
    else:
       illumination_img_1 = Get_Gauss_map(img, center_1, center_2, power, scale)
       illumination_img_2 = Get_Linear_map(img, center_1, center_2, power, scale)
       illumination_img = (illumination_img_1>>1) + (illumination_img_2>>1)

    return illumination_img
