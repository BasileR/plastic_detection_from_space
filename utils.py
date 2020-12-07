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
    L : Liste de str, indiquant les bandes pectrales à utiliser.
        exemple : L = ['B08','BO4']
    folder : Str, Indiquant le nom du dossier dans lequel chercher. Un dossier
        correspondant à une date et un lieu de capture de d'une image.
        exemple : '2019_04_18_M'

    Returns
    -------
    band_dict : Dictionnaire de np.array(), 1 array pour 1 bande spectrale
        dont les coefficients sont les réflectivités normalisées pour la bande
        spectrale considérée
        exemple : band_dict = {'B08' : array, 'B04' : 'array'}

    '''
    ## Test pour assurer le bon fonctionnement de la fonction
    if len(L) == 0  or type(L) != list:
        raise TypeError("L n'est pas une liste, ou L est vide.")
    if type(folder) != str:
        raise TypeError("folder n'est pas un str.")

    ## Obtention des chemins pour charger les bandes spectrales
    path = 'data/'+ folder
    inside_folder = os.listdir(path)

    band_dict = {}

    ## Chargement des bandes spectrales désirées
    for raw_band in inside_folder:
        for desired_band in L:
            if desired_band in raw_band:
                band_dict[desired_band] = tifffile.imread(os.path.join(path,raw_band))

    return band_dict


def get_FDI(folder):

    '''
    Parameters
    ----------
    folder : nom du dossier désiré pour les bandes spectrales

    Returns
    -------
    FDI : Calcule le FDI (Floating Debris Index) et le retourne sous forme
        d'un array de même dimension que arrays en entrée.

    '''
    ## valeurs des longueurs d'ondes de B08, B04 et B11 nécessaires au calcul
    lambda_8 = 833.8
    lambda_4 = 664.6
    lambda_11 = 1613.7

    required_band = ['B08','B11','B06']

    band_dict = get_band(required_band,folder)

    ## Test si toutes les bandes requises sont chargées dans band_dict
    for band in required_band:
        if band not in list(band_dict.keys()):
            raise ValueError('Band_dict ne contient pas les bandes requises\
                             pour le FDI, required_band = [B08,B11,B06]')
    B08 = band_dict['B08']
    B11 = band_dict['B11']
    B06 = band_dict['B06']

    return B08 - ( B06 + (B11-B06)*10*(lambda_8-lambda_4)/(lambda_11-lambda_4) )

def get_PI(folder):
    '''
    Parameters
    ----------
    folder : nom du dossier désiré pour les bandes spectrales

    Returns
    -------
    PI : Calcule le PI (Plastic Index) et le retourne sous forme
        d'un array de même dimension que arrays en entrée.
    '''
    required_band = ['B08','B04']

    band_dict = get_band(required_band,folder)

    ## Test si toutes les bandes requises sont chargées dans band_dict
    for band in required_band:
        if band not in list(band_dict.keys()):
            raise ValueError('Band_dict ne contient pas les bandes requises pour le PI, required_band = [B08,B04]')

    B08 = band_dict['B08']
    B04 = band_dict['B04']

    return B08/(B04+B08)


def get_RNDVI(folder):
    '''
    Parameters
    ----------
    folder : nom du dossier désiré pour les bandes spectrales

    Returns
    -------
    RNDVI : Calcule le RNDVI (Floating Debris Index) et le retourne sous forme
        d'un array de même dimension que arrays en entrée.
    '''   
    required_band = ['B08','B04']

    band_dict = get_band(required_band,folder)

    ## Test si toutes les bandes requises sont chargées dans band_dict
    for band in required_band:
        if band not in list(band_dict.keys()):
            raise ValueError('Band_dict ne contient pas les bandes requises \
                             pour le RNDVI, required_band = [B08,B04]')

    B08 = band_dict['B08']
    B04 = band_dict['B04']

    return (B04-B08)/(B04+B08)

def plot_index(index):
    plt.imshow(index)
    plt.colorbar()
    plt.show()

def interpolation_bicubique_multi(A, multi):
  # Effectue l'interpolation multicubique
  if (type(A) != np.ndarray):
    raise TypeError("L'argument n'est pas un ndarray enculé")
  m, n = A.shape
  B = np.zeros((m*multi,n*multi))
  for i in range(multi):
    for j in range(multi):
      B[i:B.shape[0]:multi,j:B.shape[1]:multi] = A
  return B

def interpolation_bicubique_inverse(A,multi):
  # Effectue l'interpolation multicarrée inverse
  return A[0:A.shape[0]:multi,0:A.shape[1]:multi]

def maximize(img, zoom_int):
    return interpolation_bicubique_multi(img, zoom_int)

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
    ROIs = cv2.selectROIs("Select the ROI : Click and drag the mouse (top left to bottom right) and press Enter, or c to cancel", img)
    Coords_list = list()
    #print(ROIs)
    for ROI in ROIs:
        #imgCrop = img[int(ROI[1]):int(ROI[1]+ROI[3]), int(ROI[0]):int(ROI[0]+ROI[2])]
        Coords_list.append([int(ROI[1]), int(ROI[1]+ROI[3]), int(ROI[0]), int(ROI[0]+ROI[2])])
        #cv2.imshow("Image", imgCrop)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    #print(Coords_list)
    cv2.destroyAllWindows()
    return Coords_list


def create_label(img, coords_list, label_on_source_image):
    label = np.zeros(img.shape)
    if label_on_source_image:
        label = img.copy()
    for coords in coords_list:
        label[coords[0]:coords[1], coords[2]:coords[3]] = 1
    return label

def plot_label_vs_image(image):
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
