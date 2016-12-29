#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 23:44:12 2016

@author: quien
"""

import glob;

import numpy as np;
import numpy.random as rd;

import matplotlib.image as img;
import matplotlib.pyplot as plt;

names = glob.glob("train_set/*/*.pgm");

A = None;
d = None;
for name in names:
    a = img.imread(name)/255.0;
    d = a.shape;
    a = a.reshape((a.shape[0]*a.shape[1],1));
    a -= np.mean(a);
    a /= np.linalg.norm(a)+1e-10;
    if A is None:
        A = a;
    else:
        A = np.concatenate((A,a),axis=1);
    
mu_A = np.mean(A,axis=1);

plt.imshow(mu_A.reshape(d));

A_cen = A-np.repeat(mu_A.reshape((mu_A.shape[0],1)),A.shape[1],axis=1);

sig_A = np.dot(A_cen,A_cen.T)/A.shape[1];

plt.imshow(sig_A);

E,V = np.linalg.eig(sig_A);

A_w = np.dot(np.dot(V,np.diag(1./np.sqrt(E+1e-10))),np.dot(V.T,A_cen))

D_n = 16;
D_m = 16;
D = np.dot(A_w,rd.randn(A_w.shape[1],D_n*D_m));
n_D = np.linalg.norm(D,axis=0);
D /= np.repeat(n_D.reshape((1,n_D.shape[0])),D.shape[0],axis=0);

K = 2;

for t in range(20):
    print t;
    S = np.dot(D.T,A_w);
    S_ = np.argsort(np.abs(S),axis=1);    
    S[S_>=K] = 0.0;
    D = np.dot(A_w,S.T)+D;
    n_D = np.linalg.norm(D,axis=0);
    D /= np.repeat(n_D.reshape((1,n_D.shape[0])),D.shape[0],axis=0);

f,axarr = plt.subplots(8,8);
for i in range(8):
    for j in range(8):
        axarr[i,j].imshow(D[:,D_m*i+j].reshape(d),cmap='gray');
        axarr[i,j].set_xticklabels([]);
        axarr[i,j].set_yticklabels([]);
        axarr[i,j].grid(False)
plt.show()