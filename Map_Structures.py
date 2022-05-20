#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Oliver
"""

import re
import numpy as np
import pandas as pd
import scipy.spatial
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import copy
import mdtraj as md
import os

# Save one big dictionary for every analysis

#%%
###############
## Functions ##
###############

### Add docstring
### Add comments

def get_Input(file_setup, ligand_setup=[], initial_path_proteins = ".", initial_path_ligands = ".", index = 0):
    # Get Orthogroup
    orthogroup = file_setup.loc[index,"Orthogroup"]
    # Get Identifier
    identifier = file_setup.loc[index,"New directory"]
    # Get Path Proteins
    path_proteins = os.path.join(initial_path_proteins,identifier)
    # Get Path Ligands
    path_ligands = os.path.join(initial_path_ligands,identifier)
    # Get Reference Structure    
    reference = file_setup.loc[index, "#1"][:-4]
    # Get all Files for Orthogroup and Reference
    files_proteins_raw = file_setup.loc[index].values[2:][~pd.isnull(file_setup.loc[index].values[2:])]
    # Get final File Pathes Proteins
    files_proteins = [os.path.join(path_proteins,file) for file in files_proteins_raw]
    # Check for known ligands
    if any(ligand_setup):
        # Set flag
        ligands_exist = True
        # Get Ligands raw
        files_ligands_raw = ligand_setup.loc[identifier].values[2:][~pd.isnull(ligand_setup.loc[identifier].values[2:])]
        # Get final File Pathes Ligands
        files_ligands = [os.path.join(path_ligands,file) for file in files_ligands_raw]
    else:
        # Set flag
        ligands_exist = False
        # Return empty list (skip some analysis later on)
        files_ligands = []
    return orthogroup, identifier, reference, files_proteins, files_ligands, ligands_exist

def get_Dictionary_Molecules(reference, files_proteins):
    # Define Dictonary    
    dict_molecules = {reference:[]}
    
    for file in files_proteins:
        # Generate key
        mol = re.search(r"/[0-9A-Za-z_\-]+\.pdb",file).group(0)[1:-4]
        dict_molecules.update({mol:
                               {"Molecule":open(file,"r").read().splitlines()}
                               })
    return dict_molecules
                                

def get_Dictionary_Molecular_Analysis(reference, files_proteins, dict_molecules, do_dssp=True, do_sasa=True):
    for file in files_proteins:
        # Generate key
        mol = re.search(r"/[0-9A-Za-z_\-]+\.pdb",file).group(0)[1:-4]
        if do_dssp or do_sasa:
            # Load molecule for MDTraj analysis
            molecule = md.load(file)
        # Extract, Molecule coordinates, DSSP, SASA
        if do_dssp:
            dssp = md.compute_dssp(molecule,simplified=True).reshape(-1)
            dict_molecules[mol].update({"DSSP":dssp})
        if do_sasa:
            sasa = md.shrake_rupley(molecule,mode="residue").reshape(-1)
            dict_molecules[mol].update({"SASA":sasa})
    return dict_molecules

def get_Dictionary_Proteins(reference, dict_molecules):
    for key, file in dict_molecules.items():
        # Create empty lists for different properties/atom types
        c_alpha = []
        amino_acid_type = []
        heteroatoms = []
        coordinates = []
        amino_acid_index = []
        plddt = []
        for line in file["Molecule"]:
            # Get all cordinates and indices
            if ("ATOM " in line):
                amino_acid_index.append(int(line[22:26]))
                coord = [float(el) for el in [line[30:38],line[38:46],line[46:54]]]
                coordinates.append(coord)
            # Get C_alpha, atom types and pLDDT
            if ("ATOM " in line) and ("CA " in line):
                coord = [float(el) for el in [line[30:38],line[38:46],line[46:54]]]                
                c_alpha.append(coord)
                amino_acid_type.append(line[17:20])
                plddt.append(float(line[60:66]))
            # Get Heteroatoms
            elif ("HETATM" in line) and ("HOH" not in line):
                coord = [float(el) for el in [line[30:38],line[38:46],line[46:54]]]
                heteroatoms.append(coord)
            
        # Update Dictionary
        dict_molecules[key].update({"C_alpha":c_alpha,
                                "Residues":amino_acid_type,
                                "Heteroatoms":heteroatoms,
                                "Indices":amino_acid_index,
                                "Coordinates":coordinates,
                                "pLDDT":plddt})
    return dict_molecules

def get_ligands(files_ligands):
    # Create empty list
    ligands = [] 
    for file in files_ligands:
        # Load Mapped Crystal structure (Convert .ent -> .pdb if necessary)
        molecule = open(file[:-4]+".pdb","r").read().splitlines()
        # Extract Ligand Coordinates
        for line in molecule:
            if (line.startswith("HETATM")) and (line[17:20].strip() not in ligands_to_be_excluded):
                ligands.append([float(el) for el in [line[30:38],line[38:46],line[46:54]]])
    return ligands

def get_indices_residues_ligands(dict_molecules, ligands, dist_cut_lig = 5, k = 100):
    for key, file in dict_molecules.items():
        tree_lig = scipy.spatial.cKDTree(file["Coordinates"]) 
        mapping_lig = []
        for item in ligands: 
            neighbor = tree_lig.query(item, k=k, distance_upper_bound = dist_cut_lig)
            mapping_lig.append(neighbor[1][neighbor[0]!=np.inf])
        mapping_lig = np.asarray(np.unique([el for cluster in mapping_lig for el in cluster]))
        mapping_lig_ai = np.unique(np.asarray([dict_molecules[key]["Indices"][item] for item in mapping_lig]))
        dict_molecules[key].update({"Ligand Indices":mapping_lig_ai})   
    return dict_molecules

def mapping_Proteins(dict_molecules, reference, dist_cut = 5):
    # Set reference
    ref = dict_molecules[reference]["C_alpha"]
    keys_list = [reference]
    # Build reference tree
    tree = scipy.spatial.cKDTree(ref) 
    
    # Preset mapping procedure
    mapping = np.zeros((len(dict_molecules),len(ref)),dtype="int64")-1
    mapping_dist = np.zeros_like(mapping, dtype="float64")-1
    
    # Preset counting variable
    count = 0
    
    # Get closest neighbor within 2 A
    for key, values in dict_molecules.items():
        if key != reference:      
            probe = values["C_alpha"]
            keys_list.append(key)
            for ind,item in enumerate(probe):
                neighbor = tree.query(item, k = 1, distance_upper_bound = dist_cut)
                # Check for mapping
                if neighbor[0] != np.inf:
                    mapping[count,neighbor[1]] = ind
                    mapping_dist[count,neighbor[1]] = neighbor[0]
        # Reference
        else:
            mapping[count,:] = np.arange(len(ref))
            mapping_dist[count,:] = np.zeros(len(ref))
        count += 1
        # Merge into dictionary
        dict_mapping = {"Mapping":mapping,
                        "Mapping distance":mapping_dist,
                        "Keys":keys_list}
    return dict_mapping

def write_indices_mapping(dict_molecules, dict_mapping, dist_cut = 2, output_path_data = "."):
    for ind,key in enumerate(dict_molecules.keys()):
        with open(os.path.join(output_path_data, "indices_mapping_{:.2f}_".format(dist_cut)+key+".txt"),"w") as file:
            for el in dict_mapping["Mapping"][ind,:]:
                if el != -1:
                    file.write(str(el))
                    file.write(" ")
                    
def mapping_Characteristics(dict_molecules, dict_mapping, do_dssp=True, do_sasa=True):
    # Map to amino acids and amino acids type
    mapping_aa = np.zeros_like(dict_mapping["Mapping"],dtype="U25")
    mapping_pLDDT = np.zeros_like(dict_mapping["Mapping"],dtype="float64")
    mapping_DSSP = np.zeros_like(dict_mapping["Mapping"],dtype="U25")
    mapping_SASA = np.zeros_like(dict_mapping["Mapping"],dtype="float64")-1

    count = 0
    
    for key, values in dict_molecules.items():
        mapping_aa[count,:] = np.asarray([values["Residues"][ind] if not ind == -1 else "XYZ" for ind in dict_mapping["Mapping"][count,:]])
        mapping_pLDDT[count,:] = np.asarray([values["pLDDT"][ind] if ind!=-1 else -1 for ind in dict_mapping["Mapping"][count,:]])
        if do_dssp:
            mapping_DSSP[count,:] = np.asarray([values["DSSP"][ind] if ind!=-1 else "NoMatch" for ind in dict_mapping["Mapping"][count,:]])
        else:
            mapping_DSSP[count,:] = np.asarray(["NoMatch"]*len(dict_mapping["Mapping"][count,:]))
        if do_sasa:
            mapping_SASA[count,:] = np.asarray([values["SASA"][ind] if ind!=-1 else -1 for ind in dict_mapping["Mapping"][count,:]])
        count+=1
    
    dict_mapping.update({"Amino Acid":mapping_aa,
                         "Amino Acid Type":np.vectorize(dict_atom_type.get)(mapping_aa),
                         "pLDDT":mapping_pLDDT,
                         "DSSP":mapping_DSSP,
                         "SASA":mapping_SASA})
    
    
    return dict_mapping

def mapping_colors(dict_mapping, dict_colors, dict_molecules, reference, color_match = "#808B96", bad_color = "#D0D0D0", ligands_exist = False):
    dict_mapping_colors = {}
    
    # Discrete colors
    for characteristic in ["Amino Acid","Amino Acid Type","DSSP"]:
        mapping_at_col = np.zeros_like(dict_mapping["Mapping"],dtype="int64")
        colors_cmap = []
        key_list_colors = []
        for ind,key in enumerate(dict_colors[characteristic]):
            mapping_at_col[dict_mapping[characteristic]==key] = ind
            colors_cmap.append(dict_colors[characteristic][key])
            key_list_colors.append(key)
        cmap_match = ListedColormap(colors_cmap)
        dict_mapping_colors.update({"Mapping "+characteristic:mapping_at_col,
                                    "Color keys "+characteristic:key_list_colors,
                                    "Colormap "+characteristic:cmap_match})
    
        # Reduced Maps
        mapping_at_col_red = mapping_at_col.copy()
        colors_cmap_red = colors_cmap.copy()
        key_list_colors_red = key_list_colors.copy()
        
        for i in range(1,len(mapping_at_col)):
            if characteristic == "Amino Acid Type":
                mapping_at_col_red[i,np.where((mapping_at_col[i,:]-mapping_at_col[0,:]==0)&(mapping_at_col[i,:]!=7))] = 8
            elif characteristic == "Amino Acid":
                mapping_at_col_red[i,np.where((mapping_at_col[i,:]-mapping_at_col[0,:]==0)&(mapping_at_col[i,:]!=20))] = 21
            elif characteristic == "DSSP":
                mapping_at_col_red[i,np.where((mapping_at_col[i,:]-mapping_at_col[0,:]==0)&(mapping_at_col[i,:]!=3))] = 4
        colors_cmap_red.append(color_match)
        key_list_colors_red.append("Match")
        cmap_match_red = ListedColormap(colors_cmap_red)
        dict_mapping_colors.update({"Mapping "+characteristic+" reduced":mapping_at_col_red,
                                    "Color keys "+characteristic+" reduced":key_list_colors_red,
                                    "Colormap "+characteristic+" reduced":cmap_match_red})        
    # Continuous colors
    for characteristic in ["pLDDT","SASA"]:
        cmap_masked = copy.copy(mpl.cm.get_cmap(dict_colors["Characteristic"][characteristic]))
        cmap_masked.set_under(bad_color)
        dict_mapping_colors.update({"Colormap "+characteristic:cmap_masked})
        
    # Binary
    no_mutations = np.zeros(np.shape(dict_mapping["Mapping"])[1])
    no_mutations[np.where([True if all(dict_mapping_colors["Mapping Amino Acid reduced"][1:,ind]>=20) else False for ind in range(np.shape(dict_mapping["Mapping"])[1])])] = 1
    no_mutations_type = np.zeros(np.shape(dict_mapping["Mapping"])[1])
    no_mutations_type[np.where([True if all(dict_mapping_colors["Mapping Amino Acid Type reduced"][1:,ind]>=7) else False for ind in range(np.shape(dict_mapping["Mapping"])[1])])] = 1
    cmap_bin = ListedColormap([dict_colors["No Mutations"][ind] for ind in [0,1]])
    dict_mapping_colors.update({"No Mutations":no_mutations,
                                "No Type Mutations":no_mutations_type,
                                "Colormap No Mutations":cmap_bin,
                                "Colormap No Type Mutations":cmap_bin})
    
    if ligands_exist:
        binding_site = np.zeros(np.shape(dict_mapping["Mapping"])[1])
        binding_site[dict_molecules[reference]["Ligand Indices"]-1]=1
        cmap_bin = ListedColormap([dict_colors["Binding Site"][ind] for ind in [0,1]])
        dict_mapping_colors.update({"Binding Site":binding_site,
                                    "Colormap Binding Site":cmap_bin})
    
    return dict_mapping_colors

def plot_Feature_comparison(key, dict_mapping, dict_colors, dict_boundaries, output_path=".", ligands_exist=True, plot_ligand=False, dict_molecules=None, reference=None):
    
    if plot_ligand and not (dict_molecules and reference):
        raise ValueError("dict_molecule and reference needed for plotting of binding site only.")
    
    if plot_ligand:
        size_full = 100
    else:
        size_full = 20
    
    if key not in ["Amino Acid","Amino Acid reduced","Amino Acid Type","Amino Acid Type reduced","DSSP","DSSP reduced","SASA","pLDDT"]:
        raise ValueError("Feature not known")

    fig = plt.figure()
    fig.set_size_inches(20,6)
    if ligands_exist:
        spec = fig.add_gridspec(ncols=1, nrows=9, height_ratios=[len(dict_mapping["Mapping"]),1,1,1,1,1,1,1,1], hspace=0)
        axl = fig.add_subplot(spec[7, 0])
        ax6 = fig.add_subplot(spec[8, 0])
    else:
        spec = fig.add_gridspec(ncols=1, nrows=8, height_ratios=[len(dict_mapping["Mapping"]),1,1,1,1,1,1,1], hspace=0)
        ax6 = fig.add_subplot(spec[7, 0])
        
    ax = fig.add_subplot(spec[0, 0])
    ax1 = fig.add_subplot(spec[2, 0])
    ax2 = fig.add_subplot(spec[3, 0])
    ax3 = fig.add_subplot(spec[4, 0])
    ax4 = fig.add_subplot(spec[5, 0])
    ax5 = fig.add_subplot(spec[6, 0])
    
    if key in ["Amino Acid","Amino Acid reduced","Amino Acid Type","Amino Acid Type reduced","DSSP","DSSP reduced"]:
        if plot_ligand:
            im = ax.matshow(dict_mapping_colors["Mapping "+key][:,dict_molecules[reference]["Ligand Indices"]-1], cmap = dict_mapping_colors["Colormap "+key], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key][0], vmax=dict_boundaries[key][1])
        else:
            im = ax.matshow(dict_mapping_colors["Mapping "+key], cmap = dict_mapping_colors["Colormap "+key], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key][0], vmax=dict_boundaries[key][1])
    else:
        if plot_ligand:
            im = ax.matshow(dict_mapping[key][:,dict_molecules[reference]["Ligand Indices"]-1], cmap = dict_mapping_colors["Colormap "+key], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key][0], vmax=dict_boundaries[key][1])            
        else:
            im = ax.matshow(dict_mapping[key], cmap = dict_mapping_colors["Colormap "+key], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key][0], vmax=dict_boundaries[key][1])
    
    ax.set_yticks(np.arange(len(dict_mapping["Keys"])))
    ax.set_yticklabels(dict_mapping["Keys"],fontsize=fs)
    if plot_ligand:
        ax.set_xticks(np.arange(10,np.shape(dict_molecules[reference]["Ligand Indices"])[0],10))        
    else:
        ax.set_xticks(np.arange(50,np.shape(dict_mapping["Mapping"])[1],50))

    key_s = "pLDDT"
    if plot_ligand:
        ax1.matshow(dict_mapping[key_s][0,dict_molecules[reference]["Ligand Indices"]-1].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])        
    else:
        ax1.matshow(dict_mapping[key_s][0,:].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
    ax1.set_yticks([0])
    ax1.set_yticklabels(["pLDDT"])
    ax1.set_xticks([])
    
    key_s = "DSSP"
    if plot_ligand:
        ax2.matshow(dict_mapping_colors["Mapping "+key_s][0,dict_molecules[reference]["Ligand Indices"]-1].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])        
    else:
        ax2.matshow(dict_mapping_colors["Mapping "+key_s][0,:].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
    ax2.set_yticks([0])
    ax2.set_yticklabels(["DSSP"])
    ax2.set_xticks([])

    key_s = "SASA"
    if plot_ligand:
        ax3.matshow(dict_mapping[key_s][0,dict_molecules[reference]["Ligand Indices"]-1].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])     
    else:
        ax3.matshow(dict_mapping[key_s][0,:].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
    ax3.set_yticks([0])
    ax3.set_yticklabels(["SASA"])
    ax3.set_xticks([])
    
    if ligands_exist:
        key_s = "Binding Site"
        if plot_ligand:
            axl.matshow(dict_mapping_colors[key_s][dict_molecules[reference]["Ligand Indices"]-1].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
        else:
            axl.matshow(dict_mapping_colors[key_s].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
        axl.set_yticks([0])
        axl.set_yticklabels(["Binding Site"])
        axl.set_xticks([])

    key_s = "No Mutations"
    if plot_ligand:
        ax4.matshow(dict_mapping_colors[key_s][dict_molecules[reference]["Ligand Indices"]-1].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
    else:
        ax4.matshow(dict_mapping_colors[key_s].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
    ax4.set_yticks([0])
    ax4.set_yticklabels(["No Mutations"])
    ax4.set_xticks([])
    
    key_s = "No Type Mutations"
    if plot_ligand:
        ax5.matshow(dict_mapping_colors[key_s][dict_molecules[reference]["Ligand Indices"]-1].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
    else:
        ax5.matshow(dict_mapping_colors[key_s].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
    ax5.set_yticks([0])
    ax5.set_yticklabels(["No Type Mutations"])
    ax5.set_xticks([])
    
    key_s = "Amino Acid Type"
    if plot_ligand:
        ax6.matshow(dict_mapping_colors["Mapping "+key_s][0,dict_molecules[reference]["Ligand Indices"]-1].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
    else:
        ax6.matshow(dict_mapping_colors["Mapping "+key_s][0,:].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
    ax6.set_yticks([0])
    ax6.set_yticklabels(["Amino Acid Type"])
    ax6.set_xticks([])

    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    if key in ["Amino Acid","Amino Acid reduced","Amino Acid Type","Amino Acid Type reduced","DSSP","DSSP reduced"]:
        cbar.set_ticks(np.arange(0,len(dict_mapping_colors["Color keys "+key])))
        cbar.set_ticklabels(dict_mapping_colors["Color keys "+key])
    else:
        cbar.set_label(key,rotation=90, fontsize=fs)
    
    if plot_ligand:
        plt.savefig(os.path.join(output_path,key.replace(" ","_")+"_binding_site.png"), bbox_inches="tight")
    else:
        plt.savefig(os.path.join(output_path,key.replace(" ","_")+".png"), bbox_inches="tight")
    plt.close()

def write_output(dict_molecules,dict_mapping,dict_mapping_colors, reference, output_path_data="."):
    dict_full = {"Molecules":dict_molecules,
                 "Mapping":dict_mapping,
                 "Mapping Colors":dict_mapping_colors}
    pickle.dump(dict_full, open(os.path.join(output_path_data,reference+".pkl"),"wb"))    
    
#%%
##################
## Dictionaries ##
##################

## Amino Acid Type
dict_atom_type = {"LYS":"Basic","ARG":"Basic","HIS":"Basic",
                  "GLU":"Acidic","ASP":"Acidic",
                  "TRP":"Aromatic","PHE":"Aromatic","TYR":"Aromatic",
                  "THR":"Polar","SER":"Polar","CYS":"Polar","GLN":"Polar","ASN":"Polar",
                  "ALA":"Apolar","VAL":"Apolar","LEU":"Apolar","ILE":"Apolar","MET":"Apolar",
                  "PRO":"Break",
                  "GLY":"Flexible",
                  "XYZ":"NoMatch"}

## Colors
dict_colors = {"Amino Acid":
                   {"LYS":"#154360","ARG":"#21618C","HIS":"#5499C7",
                    "GLU":"#641E16","ASP":"#CB4335",
                    "TRP":"#145A32","PHE":"#239B56","TYR":"#52BE80",
                    "THR":"#4A235A","SER":"#6C3483","CYS":"#9B59B6","GLN":"#C39BD3","ASN":"#D2B4DE",
                    "ALA":"#B9770E","VAL":"#D68910","LEU":"#F4D03F","ILE":"#F1C40F","MET":"#D4AC0D",
                    "PRO":"#1C2833",
                    "GLY":"#85929E",
                    "XYZ":"#D0D0D0"},
               "Amino Acid Type":
                   {"Basic":"#3498DB",        #Blue
                    "Acidic":"#C0392B",       #Red    
                    "Aromatic":"#28B463",     #Green
                    "Polar":"#8E44AD",        #Violet
                    "Apolar":"#E67E22",       #Orange
                    "Break":"#212F3D",        #Dark
                    "Flexible":"#F7DC6F",     #Yellow
                    "NoMatch":"#D0D0D0"},     #Gray
               "DSSP":
                   {"E":"C3",                 #Red
                    "H":"C0",                 #Blue
                    "C":"C8",                 #Yellow
                    "NoMatch":"#D0D0D0"},     #Gray
               "Binding Site":
                   {0:"w",
                    1:"k"},
               "No Mutations":
                   {0:"w",
                    1:"k"},
                "Characteristic":
                    {"pLDDT":"RdPu",
                     "SASA":"Blues"}}

# Boundaries for heatmap plots
dict_boundaries = {"Amino Acid":[-0.5, 20.5],
                   "Amino Acid reduced":[-0.5, 21.5],
                   "Amino Acid Type":[-0.5, 7.5],
                   "Amino Acid Type reduced":[-0.5, 8.5],
                   "DSSP":[-0.5,3.5],
                   "DSSP reduced":[-0.5,4.5],
                   "SASA":[0,3],
                   "pLDDT":[0,100],
                   "Binding Site":[0,1],
                   "No Mutations":[0,1],
                   "No Type Mutations":[0,1]}
    
## Ligands to be excluded
ligands_to_be_excluded = ["2HP","BA","CL","F","HG","K","LI","NA","NH4","PB","PO4","SO4","NO3","POP","ACE","AZI","HOH"]


#%%
###########
## Input ##
###########

# Distance cutoff for mapping
dist_cut = 2
# Distance cutoff for binding site detection
dist_cut_lig = 5
# Fontsize for figures
fs = 10
# Color for Matching residues (no mutation)
color_match = "#808B96"
# Color for "bad" values
bad_color = "#D0D0D0"
# Set initial path proteins
initial_path_proteins = "../Test/Proteins"
# Set initial path ligands
initial_path_ligands = "../Test/Ligands"
# Set File Mapping
file_mapping = "../Output_data/Mapping.tsv"
# Set Ligand Mapping
ligand_mapping = "../Output_data/Mapping_Ligands.tsv"
# Set output path data
output_path_data = "../Test/Output_data"
# Set output path figures
output_path = "../Test/Output"
# Set index for structure to be computed
index = 66
# Set number of neighbors to be taken into account for neighbor search
k = 100

#%%
###########
## Flags ##
###########

# Write output for mapped indices
write_output_indices = True
# Should secondary structure analysis be performed?
do_dssp = True
# Should solvent accessible surface area calculation be performed?
do_sasa = True
# Should plots be generated?
do_plots = True
# Should plots for the binding site be generated?
do_plots_binding_site = True

#%%
###########
## Setup ##
###########

# Read Input Proteins
file_setup = pd.read_csv(file_mapping,sep="\t")
# Read Input Ligands
ligand_setup = pd.read_csv(ligand_mapping,sep="\t",index_col="New directory")

#%%
####################
## Initialization ##
####################

# Get Inputs
orthogroup, identifier, reference, files_proteins, files_ligands, ligands_exist = get_Input(file_setup, ligand_setup = ligand_setup, initial_path_proteins = initial_path_proteins, initial_path_ligands = initial_path_ligands, index = index)
# Get Dictonary for molecule
dict_molecules = get_Dictionary_Molecules(reference, files_proteins)
dict_molecules = get_Dictionary_Molecular_Analysis(reference, files_proteins, dict_molecules, do_dssp=do_dssp, do_sasa=do_sasa)
dict_molecules = get_Dictionary_Proteins(reference, dict_molecules)

#%%
############
## Ligand ##
############

### Check w/ and w/o ligand
# If ligand(s) are present get potentially coordinating residues
if ligands_exist:
    ligands = get_ligands(files_ligands)
    dict_molecules = get_indices_residues_ligands(dict_molecules, ligands, dist_cut_lig = dist_cut_lig, k = k)

#%%
#############
## Mapping ##
#############

# Map proteins
dict_mapping = mapping_Proteins(dict_molecules, reference, dist_cut = dist_cut)
if write_output_indices:
    write_indices_mapping(dict_molecules, dict_mapping, dist_cut = dist_cut, output_path_data = output_path_data)
# Map residues
dict_mapping = mapping_Characteristics(dict_molecules, dict_mapping, do_dssp = do_dssp, do_sasa = do_sasa)
# Map colors
dict_mapping_colors = mapping_colors(dict_mapping, dict_colors, dict_molecules, reference, color_match = color_match, bad_color = bad_color, ligands_exist = ligands_exist)

#%%
###########
## Plots ##
###########

# Plot characteristics
for key in ["Amino Acid","Amino Acid reduced","Amino Acid Type","Amino Acid Type reduced","DSSP","DSSP reduced","SASA","pLDDT"]:
    if do_plots:
        plot_Feature_comparison(key, dict_mapping, dict_colors, dict_boundaries, output_path=output_path, ligands_exist=True)
    if do_plots_binding_site:
        plot_Feature_comparison(key, dict_mapping, dict_colors, dict_boundaries, output_path=output_path, ligands_exist=True, plot_ligand=True, dict_molecules=dict_molecules, reference=reference)

############
## Output ##
############

# Write output to file
write_output(dict_molecules, dict_mapping,dict_mapping_colors, reference, output_path_data=output_path_data)

#%%
############
## Scores ##
############

### Additional script

# Use flag
# Predefine dataframe for performace -> generate list of columns, take list of indices



