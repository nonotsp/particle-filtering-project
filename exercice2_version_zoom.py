import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import matplotlib.image as mpimg
from PIL import Image
import math
from scipy.stats import norm
from numpy.random import random
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
import os
import time

def multinomial_resample(weights):

    weights=weights.T
    cumulative_sum = np.cumsum(weights)
    #cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    #print ( np.searchsorted(cumulative_sum, random(len(weights))))
    return np.searchsorted(cumulative_sum, random(len(weights)))


def lecture_image() :

    SEQUENCE = "./videos_sequences/sequence1/"
    #charge le nom des images de la séquence
    filenames = os.listdir(SEQUENCE)
    T = len(filenames)
    #charge la premiere image dans ’im’
    tt = 0

    im=Image.open((str(SEQUENCE)+str(filenames[tt])))
    plt.imshow(im)
    
    return(im,filenames,T,SEQUENCE)

def selectionner_zone() :

    #lecture_image()
    print('Cliquer 4 points dans l image pour definir la zone a suivre.') ;
    zone = np.zeros([2,4])
 #   print(zone))
    compteur=0
    while(compteur != 4):
        res = plt.ginput(1)
        a=res[0]
        #print(type(a)))
        zone[0,compteur] = a[0]
        zone[1,compteur] = a[1]   
        plt.plot(a[0],a[1],marker='X',color='red') 
        compteur = compteur+1 

    #print(zone)
    newzone = np.zeros([2,4])
    newzone[0, :] = np.sort(zone[0, :]) 
    newzone[1, :] = np.sort(zone[1, :])
    
    zoneAT = np.zeros([4])
    zoneAT[0] = newzone[0,0]
    zoneAT[1] = newzone[1,0]
    zoneAT[2] = newzone[0,3]-newzone[0,0] 
    zoneAT[3] = newzone[1,3]-newzone[1,0] 
    #affichage du rectangle
    #print(zoneAT)
    xy=(zoneAT[0],zoneAT[1])
    rect=ptch.Rectangle(xy,zoneAT[2],zoneAT[3],linewidth=3,edgecolor='red',facecolor='None') 
    #plt.Rectangle(zoneAT[0:1],zoneAT[2],zoneAT[3])
    currentAxis = plt.gca()
    currentAxis.add_patch(rect)
    plt.show(block=False)
    plt.close()
    return(zoneAT)


def rgb2ind(im,nb) :
    #nb = nombre de couleurs ou kmeans qui contient la carte de couleur de l'image de référence
    print(im)
    image=np.array(im,dtype=np.float64)/255
    w,h,d=original_shape=tuple(image.shape)
    image_array=np.reshape(image,(w*h,d))
    image_array_sample=shuffle(image_array,random_state=0)[:1000]
    print(image_array_sample.shape)
   # print(type(image_array))
    if type(nb)==int :
        kmeans=KMeans(n_clusters=nb,random_state=0).fit(image_array_sample)
    else :
        kmeans=nb
            
    labels=kmeans.predict(image_array)
    #print(labels)
    image=recreate_image(kmeans.cluster_centers_,labels,w,h)
    #print(image)
    return(Image.fromarray(image.astype('uint8')),kmeans)

def recreate_image(codebook,labels,w,h):
    d=codebook.shape[1]
    #image=np.zeros((w,h,d))
    image=np.zeros((w,h))
    label_idx=0
    for i in range(w):
        for j in range(h):
            #image[i][j]=codebook[labels[label_idx]]*255
            image[i][j]=labels[label_idx]
            #print(image[i][j])
            label_idx+=1

    return image



def calcul_histogramme(im,zoneAT,Nb):

  #  print(zoneAT)
    box=(zoneAT[0],zoneAT[1],zoneAT[0]+zoneAT[2],zoneAT[1]+zoneAT[3])
   # print(box)
    littleim = im.crop(box)
##    plt.imshow(littleim)
##    plt.show()
    new_im,kmeans= rgb2ind(littleim,Nb)
    histogramme=np.asarray(new_im.histogram())
##  print(histogramme)
    histogramme=histogramme/np.sum(histogramme)
  #  print(new_im)
    return (new_im,kmeans,histogramme)

N=50
Nb=10
ecart_type=np.sqrt(50)
lambda_im=20
T=50
c1=300
c2=300
c3=2/100
C=np.diag([c1,c2])  

[im,filenames,T,SEQUENCE]=lecture_image()   
zoneAT=selectionner_zone()
new_im,kmeans,histo_ref=calcul_histogramme(im,zoneAT,Nb)

# Pour l'initialisation des particules
x_initial, y_initial = zoneAT[0], zoneAT[1]
width, height = zoneAT[2], zoneAT[3]

particles = np.zeros((N, 3))  # [X1, X2, X3]
particles[:, 0] = x_initial + np.random.normal(0, np.sqrt(c1), N)
particles[:, 1] = y_initial + np.random.normal(0, np.sqrt(c2), N)
particles[:, 2] = 1 + np.random.normal(0, np.sqrt(c3), N)  # Initialisation autour de 1
Neff = np.zeros(T)

# Pour chaque image
for tt in range(T):
    # Récupérer l'image courante
    current_im = Image.open(os.path.join(SEQUENCE, filenames[tt]))
    im_width, im_height = current_im.size
    
    # Les particules suivent l'équation d'évolution
    particles[:, 0] += np.random.normal(0, np.sqrt(c1), N)
    particles[:, 1] += np.random.normal(0, np.sqrt(c2), N)
    particles[:, 2] += np.random.normal(0, np.sqrt(c3), N)
    
    # Comme les dimensions de l'image varient, on s'assure que les particules restent dedans
    particles[:, 0] = np.clip(particles[:, 0], 0, im_width - width)
    particles[:, 1] = np.clip(particles[:, 1], 0, im_height - height)
    particles[:, 2] = np.clip(particles[:, 2], 0.5, 2)
    
    # Calcule de la vraisemblance puis du poids de chaque particule
    weights = np.zeros(N)
    for i in range(N):
        
        # Ici on ne réutilise pas la fonction calcul_histogramme 
        # car on veut réutiliser le kmeans initial (obtenu ligne 138)
        x, y = particles[i, 0], particles[i, 1]
        scaled_width = width * particles[i, 2]
        scaled_height = height * particles[i, 2]
        box = (x, y, x + scaled_width, y + scaled_height)
        little_im = current_im.crop(box)
        # On réutilise le kmeans du rectangle initial
        new_im_particle, _ = rgb2ind(little_im, kmeans)
        histo_particle = np.array(new_im_particle.histogram(), dtype=float)
        histo_particle /= histo_particle.sum()
        S = np.sum(np.sqrt(histo_ref * histo_particle))
        D = np.sqrt(1 - S)
        weights[i] = np.exp(-lambda_im * D**2)
    
    # Normalisation
    if np.sum(weights) == 0:
        weights = np.ones(N) / N  # On évite la division par 0 si nos poids sont trop petits
    else:
        weights /= np.sum(weights)
    
    Neff[tt] = 1 / np.sum(weights**2, axis=0)  # Calcul de Neff à chaque instant
    
    # Rééchantilloner les particules
    indices = multinomial_resample(weights)
    particles = particles[indices, :]
    
    # Estimer selon la moyenne des particules
    estimate = np.mean(particles, axis=0)
    final_width = width * estimate[2]
    final_height = height * estimate[2]
    
    # Affichage
    plt.figure()
    plt.imshow(current_im)

    # Dessin du rectangle avec la nouvelle taille
    rect = ptch.Rectangle((estimate[0], estimate[1]), 
                      final_width, 
                      final_height, 
                      linewidth=2, 
                      edgecolor='red', 
                      facecolor='none')
    plt.gca().add_patch(rect)

    # Affichage des particules (positions X/Y seulement)
    plt.scatter(particles[:, 0], particles[:, 1], c='blue', marker='x', alpha=0.3)
    plt.axis('off')
    plt.pause(0.5)
    plt.close()
plt.plot(Neff, label=f"λ = {lambda_im}")
plt.xlabel("Temps $n$")
plt.ylabel("$N_{\\text{eff}}(n)$")
plt.legend()