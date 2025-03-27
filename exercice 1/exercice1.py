

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.random import random
import time as time

N = 50
T = 50
Q = 10
R = 1

def f(x,u,k):
  return 0.5*x + 25*(x/(1+x**2)) + 8*np.cos(1.2*k) + u

def g(x,v):
  return ((x**2)/20) + v

x_init = np.random.normal(0,np.sqrt(Q))

print(x_init)

def creer_trajectoire(Q,R,x_init,T):
  vecteur_x = np.zeros(T)
  vecteur_x[0] = x_init
  for i in range (1,T):
    u = np.random.normal(0,np.sqrt(Q))
    vecteur_x[i] = f(vecteur_x[i-1], u, i)

  return vecteur_x

vecteur_x = creer_trajectoire(Q,R,x_init,T)

def creer_observation(vecteur_x,R,T):
  vecteur_y = np.zeros(T)
  for i in range (T):
    v = np.random.normal(0,np.sqrt(R))
    vecteur_y[i] = g(vecteur_x[i],v)
  return vecteur_y

vecteur_y=creer_observation(vecteur_x,R,T)
# Afficher les vraies valeurs et les observations sur un même graphique
plt.figure(figsize=(10, 6))
plt.plot(vecteur_x, label="Vraies valeurs (x)", linewidth=2)
plt.plot(vecteur_y, label="Observations (y)", linestyle='dashed', linewidth=2, color='red')
plt.xlabel("Temps")
plt.ylabel("Valeur")
plt.title("Vraies valeurs et Observations")
plt.legend()
plt.grid()
plt.show()

def multinomial_resample(weights):
    weights=weights.T
    cumulative_sum = np.cumsum(weights)
    #cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    #print ( np.searchsorted(cumulative_sum, random(len(weights))))
    return np.searchsorted(cumulative_sum, random(len(weights)))

def filtrage_particulaire(particules, Q, R, y_n,k,f):
  particules_maj = np.zeros(N)
  for i in range(N):
    #tirer selon le mélange gaussien, on tire d'abord la particule uniformément
    j = np.random.randint(0, N)
    # puis on tire selon la loi conditionnelle à cette particule
    particules_maj[i] = np.random.normal(f(particules[j],0,k),np.sqrt(Q))

  vraisemblance = np.zeros(N)
  for i in range(N):
    vraisemblance[i] = norm.pdf(y_n, (particules_maj[i]**2)/20, np.sqrt(R))
  facteur_de_normalisation = sum(vraisemblance)
  poids = vraisemblance / facteur_de_normalisation


  x_est = particules_maj.T @ poids
  indices = multinomial_resample(poids)
  particules_nouveau = np.array([particules_maj[i] for i in indices])
  

  return x_est,particules_nouveau

def filtrage(vecteur_y,Q,R,T,f,g):

  x_estimation = np.zeros(T)
  particules = np.zeros(N)
  particules = np.random.normal(0,np.sqrt(Q),N)
  # print("init particules = ", particules)
  vraisemblance = np.zeros(N)
  for i in range(N):
    vraisemblance[i] = norm.pdf(vecteur_y[0], g(particules[i],0), np.sqrt(R))
  facteur_de_normalisation = sum(vraisemblance)
  # print("init vraisemblance =", vraisemblance)
  # print("init facteur_de_normalisation =", facteur_de_normalisation)
  poids = vraisemblance / facteur_de_normalisation
  # print("init poids =", poids, "sum poids", sum(poids))
  # ok jusqu'ici
  x_est = particules.T @ poids
  # print("init x_est =", x_est)

  indices = multinomial_resample(poids)
  particules_nouveau = np.array([particules[i] for i in indices])
  x_estimation[0] = x_est

  for i in range(1,T):
    x_est,particules_nouveau = filtrage_particulaire(particules_nouveau,Q,R,vecteur_y[i],i,f)
    x_estimation[i] = x_est


  return x_estimation

t = time.time()
x_estimation = filtrage(vecteur_y,Q,R,T,f,g)
ecoule = time.time() - t
print("temps d'execution = ",ecoule)
plt.figure(figsize=(12, 6))

# Tracé de la vraie trajectoire
plt.plot(vecteur_x, label="Vraie Trajectoire (x)", color='blue', linewidth=2)

# Tracé des observations (en pointillés)
plt.plot(vecteur_y, label="Trajectoire Observée (y)", color='red', linestyle='dashed', linewidth=1)

# Tracé de la trajectoire estimée
plt.plot(x_estimation, label="Trajectoire Estimée (x_est)", color='orange', linewidth=2)

# Ajout des détails
plt.xlabel("Temps")
plt.ylabel("Valeur")
plt.title("Comparaison des trajectoires")
plt.legend()
plt.grid()
plt.show()

def erreur_quadratique(vecteur_x,x_estimation):
  erreur = (vecteur_x - x_estimation).T @ (vecteur_x - x_estimation)
  return erreur/T
print(erreur_quadratique(vecteur_x,x_estimation))