#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Oliver Lemke
"""

import re
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Define Input
files = ["../Data/Model/H.pdb","../Data/Model/S.pdb","../Data/Model/M.pdb"]
# Define Reference
reference = "H"
# Define color codes for Amino acids
dict_colors_aa = {"LYS":"#154360","ARG":"#21618C","HIS":"#5499C7",
                  "GLU":"#641E16","ASP":"#CB4335",
                  "TRP":"#145A32","PHE":"#239B56","TYR":"#52BE80",
                  "THR":"#4A235A","SER":"#6C3483","CYS":"#9B59B6","GLN":"#C39BD3","ASN":"#D2B4DE",
                  "ALA":"#B9770E","VAL":"#D68910","LEU":"#F4D03F","ILE":"#F1C40F","MET":"#D4AC0D",
                  "PRO":"#1C2833",
                  "GLY":"#85929E",
                  "XYZ":"#D0D0D0"}
dict_at = {"LYS":"Basic","ARG":"Basic","HIS":"Basic",
           "GLU":"Acidic","ASP":"Acidic",
           "TRP":"Aromatic","PHE":"Aromatic","TYR":"Aromatic",
           "THR":"Polar","SER":"Polar","CYS":"Polar","GLN":"Polar","ASN":"Polar",
           "ALA":"Apolar","VAL":"Apolar","LEU":"Apolar","ILE":"Apolar","MET":"Apolar",
           "PRO":"Break",
           "GLY":"Flexible",
           "XYZ":"NoMatch"}
dict_colors_at = {"Basic":"#3498DB",        #Blue
                  "Acidic":"#C0392B",       #Red    
                  "Aromatic":"#28B463",     #Green
                  "Polar":"#8E44AD",        #Violet
                  "Apolar":"#E67E22",       #Orange
                  "Break":"#212F3D",        #Dark
                  "Flexible":"#F7DC6F",     #Yellow
                  "NoMatch":"#D0D0D0"}      #Gray
dist_cut = 2
dist_cut_lig = 5
fs = 25
color_match = "#808B96"

# Read in Molecules
dict_mol = {}
for file in files:
    dict_mol.update({re.search(r"/[0-9A-Za-z_]+\.pdb",file).group(0)[1:-4]:open(file,"r").read().splitlines()})

# Get CA coordinates ##### GROUP INTO ONE COMMON DICTIONARY!!!!!
dict_CA = {}
dict_residue = {}
dict_HETATM = {}
dict_coords = {}
dict_aa_index = {}
dict_pLDDT = {}
for key, file in dict_mol.items():
    ca = []
    at = []
    ha = []
    co = []
    ai = []
    for line in file:
        if ("ATOM " in line):
            ai.append(int(line[22:26]))
            coord = [float(el) for el in line[30:54].split(" ") if not el == ""]
            co.append(coord)
        if ("ATOM " in line) and ("CA " in line):
            coord = [float(el) for el in line[30:54].split(" ") if not el == ""]
            ca.append(coord)
            at.append(line[17:20])
        elif ("HETATM" in line) and ("HOH" not in line):
            coord = [float(el) for el in line[30:54].split(" ") if not el == ""]
            ha.append(coord)
            
    dict_CA.update({key:np.asarray(ca)})
    dict_residue.update({key:at})
    dict_HETATM.update({key:ha})
    dict_aa_index.update({key:ai})
    dict_coords.update({key:co})

#%%

# Get reference coordinates    
ref = dict_CA[reference]
keys_list = [reference]
tree = scipy.spatial.cKDTree(ref) 

# Preset mapping procedure
mapping = np.zeros((len(ref),len(dict_CA)),dtype="int64")-1
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
            neighbor = tree.query(item, k = 1, distance_upper_bound = dist_cut)
            if neighbor[0] != np.inf:
                mapping[neighbor[1],count] = ind

for ind,key in enumerate(dict_CA.keys()):
    with open("../Output_data/indices_mapping_{:.2f}_".format(dist_cut)+key+".txt","w") as file:
        for el in mapping[:,ind]:
            if el != -1:
                file.write(str(el))
                file.write(" ")

mapping_aa = np.zeros_like(mapping,dtype="U25")
count = 0

for key, values in dict_residue.items():
    mapping_aa[:,count] = np.asarray([values[ind] if not ind == -1 else "XYZ" for ind in mapping[:,count]])
    count+=1
    
mapping_at = np.vectorize(dict_at.get)(mapping_aa)

mapping_at_col = np.zeros_like(mapping,dtype="U25")

colors_cmap = []
for ind,key in enumerate(dict_colors_at):
    mapping_at_col[mapping_at==key] = ind
    colors_cmap.append(dict_colors_at[key])

mapping_at_col = np.int_(mapping_at_col).T
mapping_at_col_red = np.copy(mapping_at_col)
for i in range(1,len(mapping_at_col)):
    mapping_at_col_red[i,np.where((mapping_at_col[i,:]-mapping_at_col[0,:]==0)&(mapping_at_col[i,:]!=7))] = 8

colors_cmap.append(color_match)
cmap_match = ListedColormap(colors_cmap)
#%%       
#cmap_at = ListedColormap(colors_cmap)

fig,ax = plt.subplots()
fig.set_size_inches(20,6)
ax.matshow(mapping_at_col, cmap = cmap_match, aspect = np.shape(mapping_at_col)[1]/20, vmin=0, vmax=8)
ax.set_yticks(np.arange(len(dict_residue)))
ax.set_yticklabels([key for key in dict_residue.keys()],fontsize=fs)
ax.set_xticks(np.arange(50,len(mapping_at),50))
plt.savefig("../Output/test.png")

fig,ax = plt.subplots()
fig.set_size_inches(20,6)
ax.matshow(mapping_at_col_red, cmap = cmap_match, aspect = np.shape(mapping_at_col)[1]/20, vmin=0, vmax=8)
ax.set_yticks(np.arange(len(dict_residue)))
ax.set_yticklabels([key for key in dict_residue.keys()],fontsize=fs)
ax.set_xticks(np.arange(50,len(mapping_at),50))
plt.savefig("../Output/test_2.png")
#%%

ref_lig = dict_HETATM[reference]
dict_mapping_lig = {}

for key, probe in dict_coords.items():
    tree_lig = scipy.spatial.cKDTree(probe) 
    mapping_lig = []
    for item in ref_lig: 
        neighbor = tree_lig.query(item, k=100, distance_upper_bound = dist_cut_lig)
        mapping_lig.append(neighbor[1][neighbor[0]!=np.inf])
    mapping_lig = np.asarray(np.unique([el for cluster in mapping_lig for el in cluster]))
    mapping_lig_ai = np.unique(np.asarray([dict_aa_index[key][item] for item in mapping_lig]))

    dict_mapping_lig.update({key:mapping_lig_ai})

#### Be careful fo index shift!!! resid starts at 1

fig,ax = plt.subplots()
fig.set_size_inches(20,6)
ax.matshow(mapping_at_col[:,dict_mapping_lig[reference]-1], cmap = cmap_match, aspect = np.shape(mapping_at_col)[1]/20, vmin=0, vmax=8)
ax.set_yticks(np.arange(len(dict_residue)))
ax.set_yticklabels([key for key in dict_residue.keys()],fontsize=fs)
ax.set_xticks(np.arange(10,len(dict_mapping_lig[reference]),10))
plt.savefig("../Output/test_3.png")

fig,ax = plt.subplots()
fig.set_size_inches(20,6)
ax.matshow(mapping_at_col_red[:,dict_mapping_lig[reference]-1], cmap = cmap_match, aspect = np.shape(mapping_at_col)[1]/20, vmin=0, vmax=8)
ax.set_yticks(np.arange(len(dict_residue)))
ax.set_yticklabels([key for key in dict_residue.keys()],fontsize=fs)
ax.set_xticks(np.arange(10,len(dict_mapping_lig[reference]),10))
plt.savefig("../Output/test_4.png")

#%%
###DSSP


# How much is mapping (percentage)
# How Avergare distance of mapping
# How much of secondary structure is recovered (percentage)
# Mutation ratio for every pair of base, acid, polar, apolar, aromatic (e.g. base-acid -> 10 combinations)
# Weighted scores (incorparate pIDDT/pIDDT-cutoff)
# Mutation rate binding site
# How many scores above .9
# Percentage defined DSSP (H,E)

# DSSP
# Residue properties
# RMSD for aligned strucutres

        