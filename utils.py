# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:38:16 2020

@author: Gaspard Lemerle et Basile Rousse
"""

import tifffile
import numpy as np
import matplotlib.pyplot as plt
import spectral
import os
import glob
import cv2


def get_band(L,folder):
    '''
    Parameters
    ----------
    L : string list pointing to the spectral bands to use.
        example : L = ['B08','BO4','True_color']
        'True_color' is the RGB image
    folder : String pointing to the name of the folder to search. Each folder
        have the spectral bands of an image for a given date and place
        exemple : '2019_04_18_M'

    Returns
    -------
    band_dict : dictionnary of np.array(), 1 array pour 1 spectral band
        where coeff are  normalizaed reflectivies for the considered spectral
        bands
        exemple : band_dict = {'B08' : array, 'B04' : 'array'}

    '''
    ## Test of the parameters
    if len(L) == 0  or type(L) != list:
        raise TypeError("L is not a list, or is empty")
    if type(folder) != str:
        raise TypeError("folder is not a string")

    ## get paths to load spectral bands
    path = 'data/'+ folder
    inside_folder = os.listdir(path)

    band_dict = {}

    ## load wanted spectral bands
    for raw_band in inside_folder:
        for desired_band in L:
            if desired_band in raw_band:
                band_dict[desired_band] = tifffile.imread(os.path.join(path,raw_band))

    ## add the name of the folder to the dict
    band_dict['folder'] = folder

    return band_dict


def get_FDI(folder):

    '''
    Parameters
    ----------
    folder : name of the target folder (date,place) to find spectral bands

    Returns
    -------
    FDI : Calculate FDI (Floating Debris Index) and returns it as an array with
    the same size of the spectral bands.

    '''
    ## Some wavelenghts that are required (B08,B04,B11)
    lambda_8 = 833.8
    lambda_4 = 664.6
    lambda_11 = 1613.7

    required_band = ['B08','B11','B06']

    band_dict = get_band(required_band,folder)

    ## Test if every band is loaded in the dict
    for band in required_band:
        if band not in list(band_dict.keys()):
            raise ValueError('Band_dict does not contain required band for FDI\
                             required_band = [B08,B11,B06]')
    ## load bands
    B08 = band_dict['B08']
    B11 = band_dict['B11']
    B06 = band_dict['B06']

    return B08 - ( B06 + (B11-B06)*10*(lambda_8-lambda_4)/(lambda_11-lambda_4) )

def get_PI(folder):
    '''
    Parameters
    ----------
    folder : name of the target folder (date,place) to find spectral bands

    Returns
    -------
    FDI : Calculate PI (Plastic Index) and returns it as an array with
    the same size of the spectral bands.

    '''
    required_band = ['B08','B04']

    band_dict = get_band(required_band,folder)

    ## Test if every band is loaded in the dict
    for band in required_band:
        if band not in list(band_dict.keys()):
            raise ValueError('Band_dict does not contain required band for PI\
                             required_band = [B08,B04]')

    ## load bands
    B08 = band_dict['B08']
    B04 = band_dict['B04']

    return B08/(B04+B08)


def get_RNDVI(folder):
    '''
    Parameters
    ----------
    folder : name of the target folder (date,place) to find spectral bands

    Returns
    -------
    FDI : Calculate RNDVI and returns it as an array with
    the same size of the spectral bands.

    '''
    required_band = ['B08','B04']

    band_dict = get_band(required_band,folder)

    ## Test if every band is loaded in the dict
    for band in required_band:
        if band not in list(band_dict.keys()):
            raise ValueError('Band_dict does not contain required band for RNDVI\
                             required_band = [B08,B04]')

    ## load bands
    B08 = band_dict['B08']
    B04 = band_dict['B04']

    return (B04-B08)/(B04+B08)

def plot_index(index):
    """
    Parameters:
    ----------
    index : numpy array representing an image (2D or 3D, i.e. colored)

    Returns
    -------
    None


    Takes the index array and plots it using matplotlib.pyplot library,
    with a colorbar.
    """
    plt.figure()
    plt.imshow(index)
    plt.colorbar()
    plt.show()


def interpolation_multicubique(A, multi):
    """
    Parameters:
    ----------
    A : numpy array, two-dimensional (m, n) or three-dimensional (m, n, t)
    multi : integer, representing the coefficient of the interpolation

    Returns
    -------
    B : numpy array, with its first two dimensions aumgented by the coefficient
    multi -> B of dimension (m*multi, n*multi) or (m*multi, n*multi, t) if 3D
    """
    
    if (type(A) != np.ndarray):
        raise TypeError("L'argument n'est pas un ndarray")
    if (len(A.shape) == 2):
        m, n = A.shape
        B = np.zeros((m*multi,n*multi))
        for i in range(multi):
            for j in range(multi):
                 B[i:B.shape[0]:multi,j:B.shape[1]:multi] = A
        return B
    elif(len(A.shape) == 3):
        m, n, t = A.shape
        B = np.zeros((m*multi,n*multi,t))
        for k in range(t):
            for i in range(multi):
                for j in range(multi):
                     B[i:B.shape[0]:multi,j:B.shape[1]:multi,k] = A[:,:,k]
        return B


def interpolation_bicubique_inverse(A,multi):
    # Effectue l'interpolation multicarrée inverse
    if (len(A.shape) == 2):
        return A[0:A.shape[0]:multi,0:A.shape[1]:multi]
    elif (len(A.shape) == 3):
        return A[0:A.shape[0]:multi,0:A.shape[1]:multi,:]


def maximize(img, zoom_int):
    return interpolation_multicubique(img, zoom_int)

def minimize(img, zoom_int):
    return interpolation_bicubique_inverse(img, zoom_int)

def createSel(img):
    '''
    Parameters
    ----------
    img : path to the image on wich to perform the ROI selection

    Returns
    -------
    imgCrop : the cropped image, as an array (cv2 or numpy)
    Coords : the coordinates of the selection as follows (row, row+rowheight, col, col+colwidth)
    '''
    #cv2.startWindowThread()
    #img = cv2.imread(img)
    #band_dict = get_band([band_id],folder)
    #img = band_dict[band_id]
    ROIs = cv2.selectROIs("Select the ROI : Click and drag the mouse (top left to bottom right) and press Enter, or c to cancel", img)
    coords_list = list()
    #print(ROIs)
    for ROI in ROIs:
        #imgCrop = img[int(ROI[1]):int(ROI[1]+ROI[3]), int(ROI[0]):int(ROI[0]+ROI[2])]
        coords_list.append([int(ROI[1]), int(ROI[1]+ROI[3]), int(ROI[0]), int(ROI[0]+ROI[2])])
        #cv2.imshow("Image", imgCrop)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    #print(Coords_list)
    cv2.destroyAllWindows()

    #label = create_label(img, coords_list, False)

    #if save:
    #    path = save_label(label, band_dict['folder'])
    print(coords_list)
    return coords_list


def create_label(img, coords_list, label_on_source_image = False):
    label = np.zeros(img.shape)
    if label_on_source_image:
        label = img.copy()
    for coords in coords_list:
        label[coords[0]:coords[1], coords[2]:coords[3]] = 1
    return label

def save_label(label, folder):
    '''
    Params:
    label : numpy array of zeros and ones, 2-dimensional or 3-dimensional, representing the label
    folder : string, origin folder name, used to name the labeled image

    Returns:
    path: string, path to the saved image
    '''
    path = "./data/label/{}_{}.tif".format(folder,'label')
    tifffile.imsave(path,label)
    return path

def plot_label_vs_image(image, save = True):
    '''
    Parameters
    ----------
    image : image as an array (cv2 or numpy)

    Returns
    ----------
    label : numpy array representing the label (1 for plastic and 0 for other)
    '''
    zoom_int = 9
    image_zoomed = maximize(image, zoom_int)

    coords_list = createSel(image_zoomed)
    label = create_label(image_zoomed, coords_list, False)

    label_dezoomed = minimize(label, zoom_int)

    plt.figure(figsize=(15,15))
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.subplot(2, 2, 2)
    plt.imshow(label_dezoomed)
    plt.subplot(2, 2, 3)
    plt.imshow(label_dezoomed)
    plt.subplot(2, 2, 4)
    plt.imshow(image+label_dezoomed)
    plt.show()

    return label

def create_plot_save_label(origin_folder, band_id = ['B04'],zoom_int = 1, save = True, index = False):
    band_dict = get_band(band_id,origin_folder)
    img = band_dict[band_id[0]]
    if index:
        img = get_FDI(origin_folder)
        print("FDI")
    image_zoomed = maximize(img, zoom_int)
    coords_list = createSel(image_zoomed)
    label = create_label(image_zoomed, coords_list, False)
    label_dezoomed = minimize(label, zoom_int)
    plt.figure(figsize=(15,15))
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.subplot(2, 2, 2)
    plt.imshow(label_dezoomed)
    plt.subplot(2, 2, 3)
    plt.imshow(label_dezoomed)
    plt.subplot(2, 2, 4)
    plt.imshow(img+label_dezoomed)
    plt.show()

    path = 'no_path'
    if save:
        save2 = input("save : T/F (can overwrite data)")
        path = save_label(label_dezoomed, band_dict['folder']) if (save2 == 'T') else 'no_path'
    return path
