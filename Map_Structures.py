#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Oliver Lemke
"""

import re
import numpy as np
import scipy

# Define Input
files = ["../Data/Model/H.pdb","../Data/Model/S.pdb","../Data/Model/M.pdb"]
# Define Reference
reference = "H"

# Read in Molecules
dict_mol = {}
for file in files:
    dict_mol.update({re.search(r"/[0-9A-Za-z_]+\.pdb",file).group(0)[1:-4]:open(file,"r").read().splitlines()})

# Get CA coordinates
dict_CA = {}
for key, file in dict_mol.items():
    ca = []
    for line in file:
        if ("ATOM " in line) and ("CA " in line):
            coord = [float(el) for el in line[30:54].split(" ") if not el == ""]
            ca.append(coord)
    dict_CA.update({key:np.asarray(ca)})

# Get reference coordinates    
ref = dict_CA[reference]
keys_list = [reference]
tree = scipy.spatial.cKDTree(ref) 

# Preset mapping procedure
mapping = np.zeros((len(ref),len(dict_CA)))-1
mapping[:,0] = np.arange(len(ref))

# Preset counting variable
count = 0

# Get closest neighbor within 1.5 A
for key, values in dict_CA.items():
    if key != reference:      
        count += 1
        probe = values
        keys_list.append(key)
        for ind,item in enumerate(probe):
            neighbor = tree.query(item, k = 1, distance_upper_bound = 1.5)
            if neighbor[0] != np.inf:
                mapping[neighbor[1],count] = ind
                
                
        