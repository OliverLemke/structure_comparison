#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Oliver Lemke
"""

import re
import os
import copy
import pickle
import numpy as np
import pandas as pd
import scipy.spatial
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import mdtraj as md
import igraph as ig
import leidenalg as la

#%%
###############
## Functions ##
###############

### Add Plotting indices (manual indices in reference) for subplots (like binding site)
### Normalize SASA with G-X-G dictionary

def get_Input(file_setup, ligand_setup=[], initial_path_proteins = ".", initial_path_ligands = ".", index = 0):
    """
    Generates Input for all other functions and automatisation. Examples for file_setup and ligand_setup will be added later.

    Parameters
    ----------
    file_setup : pd.DataFrame
        DataFrame containing the idnetifier, the reference structure files and all related structure files.
    ligand_setup : pd.DataFrame, optional
        DataFrame containing ligand structures fitted to the reference. The default is [].
    initial_path_proteins : str, optional
        Path for protein structure files. The default is ".".
    initial_path_ligands : str, optional
        Path for ligand structure files. The default is ".".
    index : int, optional
        Index for the row in file_setup used for analysis. The default is 0.

    Returns
    -------
    group : str
        Group name.
    identifier : str
        Identifier for the group.
    reference : str
        Reference file.
    files_proteins : list of str
        List of all protein structures.
    files_ligands : list of str
        List of all ligand structures.
    ligands_exist : bool
        True if at least one ligand exists.

    """
    # Get Orthogroup
    group = file_setup.loc[index,"Orthogroup"]
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
        try:
            # Set flag
            ligands_exist = True
            # Get Ligands raw
            files_ligands_raw = ligand_setup.loc[identifier].values[2:][~pd.isnull(ligand_setup.loc[identifier].values[2:])]
            # Get final File Pathes Ligands
            files_ligands = [os.path.join(path_ligands,file) for file in files_ligands_raw]
        except:
            # Set flag
            ligands_exist = False
            files_ligands = []
    else:
        # Set flag
        ligands_exist = False
        # Return empty list (skip some analysis later on)
        files_ligands = []
    return group, identifier, reference, files_proteins, files_ligands, ligands_exist

def get_Dictionary_Molecules(reference, files_proteins):
    """
    

    Parameters
    ----------
    reference : str
        Reference file.
    files_proteins : list of str
        List of all protein structures.

    Returns
    -------
    dict_molecules : dict
        Dictonary containing structural properties.

    """
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
    """
    Update molecule dictionary for structural features

    Parameters
    ----------
    reference : str
        Reference file.
    files_proteins : list of str
        List of all protein structures.
    dict_molecules : dict
        Dictonary containing structural properties.
    do_dssp : bool, optional
        Perform secondary structure prediction. The default is True.
    do_sasa : TYPE, optional
        Perform accessible surface area prediction. The default is True.

    Returns
    -------
    dict_molecules : dict
        Dictonary containing structural properties updated for structural features.

    """
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
    """
    Update dictionary for selections of atoms.

    Parameters
    ----------
    reference : str
        Reference file.
    dict_molecules : dict
        Dictonary containing structural properties.
    Returns
    -------
    dict_molecules : dict
        Dictonary containing structural properties updated for selections.

    """
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

def get_ligands(files_ligands, ligands_to_be_excluded, ligands_exist=True):
    """
    Extract ligand coordinates from file.

    Parameters
    ----------
    files_ligands : list of str
        List of all ligand structures.
    ligands_to_be_excluded : list of str
        List of ligands to be excluded.
    ligands_exist : bool
        True if at least one ligand exists. The default is True.

    Returns
    -------
    ligands : list of arr
        Ligand coordinates.
    ligands_exist : bool
        True if ligands still exist after filtering.

    """
    # Create empty list
    ligands = [] 
    for file in files_ligands:
        # Load Mapped Crystal structure (Convert .ent -> .pdb if necessary)
        molecule = open(file[:-4]+".pdb","r").read().splitlines()
        # Extract Ligand Coordinates
        for line in molecule:
            if (line.startswith("HETATM")) and (line[17:20].strip() not in ligands_to_be_excluded):
                ligands.append([float(el) for el in [line[30:38],line[38:46],line[46:54]]])
    # Check whether ligands could be extracted (due to excluded ligands)
    if not ligands:
        ligands_exist = False
    return ligands, ligands_exist

def get_indices_residues_ligands(dict_molecules, ligands, dist_cut_lig = 5, k = 100):
    """
    Get residue indices for residues in close contact to ligands. Updates dict_molecules.

    Parameters
    ----------
    dict_molecules : dict
        Dictonary containing structural properties.
    ligands : list of arr
        Ligand coordinates.
    dist_cut_lig : float, optional
        Distance cutoff for neighbor detection. The default is 5.
    k : int, optional
        Number of nearest neighbors to be detected. The default is 100.

    Returns
    -------
    dict_molecules : dict
        Dictonary containing protein properties updated for residues close to ligands.

    """
    for key, file in dict_molecules.items():
        # Setup tree for protein
        tree_lig = scipy.spatial.cKDTree(file["Coordinates"]) 
        mapping_lig = []
        # Check for neighboring w.r.t. ligands
        for item in ligands: 
            neighbor = tree_lig.query(item, k = k, distance_upper_bound = dist_cut_lig) ####test query_ball_point (removes k) or query_ball_tree (maybe faster)
            mapping_lig.append(neighbor[1][neighbor[0]!=np.inf])
        # Get unique neighbors
        mapping_lig = np.asarray(np.unique([el for cluster in mapping_lig for el in cluster]))
        # Get unique amino acid indices
        mapping_lig_ai = np.unique(np.asarray([dict_molecules[key]["Indices"][item] for item in mapping_lig]))
        # Update dictionary
        dict_molecules[key].update({"Ligand Indices":mapping_lig_ai})   
    return dict_molecules

def mapping_Proteins(dict_molecules, reference, dist_cut = 2):
    """
    Maps structures onto the reference structure (1:1 Assignment).

    Parameters
    ----------
    dict_molecules : dict
        Dictonary containing structural properties.
    reference : str
        Reference file.
    dist_cut : float, optional
        Distance cutoff for mapping. The default is 2.

    Returns
    -------
    dict_mapping : dict
        Dictionary containg all mapping information with respect to the reference structure.

    """
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
    """
    Writes mapping indices to disc for external use.

    Parameters
    ----------
    dict_molecules : dict
        Dictonary containing structural properties.
    dict_mapping : dict
        Dictionary containg all mapping information with respect to the reference structure.
    dist_cut : float, optional
        Distance cutoff for mapping. The default is 2.
    output_path_data : str, optional
        Output path for the written file. The default is ".".

    Returns
    -------
    None.

    """
    # Write indices for mapping to file for plotting
    for ind,key in enumerate(dict_molecules.keys()):
        with open(os.path.join(output_path_data, "indices_mapping_{:.2f}_".format(dist_cut)+key+".txt"),"w") as file:
            for el in dict_mapping["Mapping"][ind,:]:
                if el != -1:
                    file.write(str(el))
                    file.write(" ")
                    
def mapping_Characteristics(dict_molecules, dict_mapping, dict_atom_type, do_dssp=True, do_sasa=True):
    """
    Maps characteristics of the protein structures to the reference.

    Parameters
    ----------
    dict_molecules : dict
        Dictonary containing structural properties.
    dict_mapping : dict
        Dictionary containg all mapping information with respect to the reference structure.
    dict_atom_type : dict
         Assignment of each amino acid to a property.
    do_dssp : bool, optional
        True if secondary structure prediction was performed. The default is True.
    do_sasa : TYPE, optional
        True if accessible surface area prediction was performed. The default is True.

    Returns
    -------
    dict_mapping : dict
        Dictionary containg all mapping information with respect to the reference structure updated for structure characteristcs.

    """
    # Map to amino acids, amino acids type, pLDDT, DSSP and SASA
    mapping_aa = np.zeros_like(dict_mapping["Mapping"],dtype="U25")
    mapping_pLDDT = np.zeros_like(dict_mapping["Mapping"],dtype="float64")
    mapping_DSSP = np.zeros_like(dict_mapping["Mapping"],dtype="U25")
    mapping_SASA = np.zeros_like(dict_mapping["Mapping"],dtype="float64")-1
    
    # Set counter
    count = 0
    
    # Transfer molecule information to mapped indices
    for key, values in dict_molecules.items():
        mapping_aa[count,:] = np.asarray([values["Residues"][ind] if not ind == -1 else "XYZ" for ind in dict_mapping["Mapping"][count,:]])
        mapping_pLDDT[count,:] = np.asarray([values["pLDDT"][ind] if ind!=-1 else -1 for ind in dict_mapping["Mapping"][count,:]])
        if do_dssp:
            mapping_DSSP[count,:] = np.asarray([values["DSSP"][ind] if ind!=-1 else "NoMap" for ind in dict_mapping["Mapping"][count,:]])
        else:
            mapping_DSSP[count,:] = np.asarray(["NoMap"]*len(dict_mapping["Mapping"][count,:]))
        if do_sasa:
            mapping_SASA[count,:] = np.asarray([values["SASA"][ind] if ind!=-1 else -1 for ind in dict_mapping["Mapping"][count,:]])
        count+=1
    # Update dictionary
    dict_mapping.update({"Amino Acid":mapping_aa,
                         "Amino Acid Type":np.vectorize(dict_atom_type.get)(mapping_aa),
                         "pLDDT":mapping_pLDDT,
                         "DSSP":mapping_DSSP,
                         "SASA":mapping_SASA})
    
    
    return dict_mapping

def mapping_colors(dict_mapping, dict_colors, dict_molecules, reference, color_match = "#808B96", bad_color = "#D0D0D0", ligands_exist = False):
    """
    

    Parameters
    ----------
    dict_mapping : dict
        Dictionary containg all mapping information with respect to the reference structure.
    dict_colors : dict
        Colors for amino acids, amino acid types, DSSP, Binding site, Mutation sites. Colormaps for SASA and pLDDT.
    dict_molecules : dict
        Dictonary containing structural properties.
    reference : str
        Reference file.
    color_match : str, optional
        Hex color code for matching amino acids. The default is "#808B96".
    bad_color : str, optional
        Hex color code for not mapping amino acids. The default is "#D0D0D0".
    ligands_exist : bool, optional
        True if at least one ligand exists. The default is False.

    Returns
    -------
    dict_mapping_colors : dict
        Dictionary containg colors for mapping.
    ligands_exist : bool
        True if at least one ligand exists that could be mapped to the protein.

    """
    # Set dictionary for mapping to colors
    dict_mapping_colors = {}
    
    ### Discrete colors ###
    for characteristic in ["Amino Acid","Amino Acid Type","DSSP"]:
        # Set output arrays/lists
        mapping_at_col = np.zeros_like(dict_mapping["Mapping"],dtype="int64")
        colors_cmap = []
        key_list_colors = []
        # Map to color scheme
        for ind,key in enumerate(dict_colors[characteristic]):
            # Dicrete property
            mapping_at_col[dict_mapping[characteristic]==key] = ind
            # Colormap
            colors_cmap.append(dict_colors[characteristic][key])
            # Keys
            key_list_colors.append(key)
        # Generate Colormap
        cmap_match = ListedColormap(colors_cmap)
        # Update dictionary
        dict_mapping_colors.update({"Mapping "+characteristic:mapping_at_col,
                                    "Color keys "+characteristic:key_list_colors,
                                    "Colormap "+characteristic:cmap_match})
    
        ### Reduced Maps (Search for matching and non-matching residues w.r.t. reference) ###
        mapping_at_col_red = mapping_at_col.copy()
        colors_cmap_red = colors_cmap.copy()
        key_list_colors_red = key_list_colors.copy()
        # Get matching residues
        for i in range(1,len(mapping_at_col)):
            if characteristic == "Amino Acid Type":
                mapping_at_col_red[i,np.where((mapping_at_col[i,:]-mapping_at_col[0,:]==0)&(mapping_at_col[i,:]!=7))] = 8
            elif characteristic == "Amino Acid":
                mapping_at_col_red[i,np.where((mapping_at_col[i,:]-mapping_at_col[0,:]==0)&(mapping_at_col[i,:]!=20))] = 21
            elif characteristic == "DSSP":
                mapping_at_col_red[i,np.where((mapping_at_col[i,:]-mapping_at_col[0,:]==0)&(mapping_at_col[i,:]!=3))] = 4
        # Colormap
        colors_cmap_red.append(color_match)
        # Keys
        key_list_colors_red.append("Match")
        # Generate Colormap
        cmap_match_red = ListedColormap(colors_cmap_red)
        # Update dictionary
        dict_mapping_colors.update({"Mapping "+characteristic+" reduced":mapping_at_col_red,
                                    "Color keys "+characteristic+" reduced":key_list_colors_red,
                                    "Colormap "+characteristic+" reduced":cmap_match_red})        
    ### Continuous colors ###
    for characteristic in ["pLDDT","SASA"]:
        # Get colormap
        cmap_masked = copy.copy(mpl.cm.get_cmap(dict_colors["Characteristic"][characteristic]))
        # Mask non-mapping residues
        cmap_masked.set_under(bad_color)
        # Update dictionary
        dict_mapping_colors.update({"Colormap "+characteristic:cmap_masked})
        
    ### Binary ###
    # Get output array
    no_mutations = np.zeros(np.shape(dict_mapping["Mapping"])[1])
    # Search for residues with no mutations (AA)
    no_mutations[np.where([True if (all(dict_mapping_colors["Mapping Amino Acid reduced"][1:,ind]>=20)) and not (all(dict_mapping_colors["Mapping Amino Acid reduced"][1:,ind]==20)) else False for ind in range(np.shape(dict_mapping["Mapping"])[1])])] = 1
    # Remove overall non-mapping residues
    no_mutations[np.where([True if (all(dict_mapping_colors["Mapping Amino Acid reduced"][1:,ind]==20)) else False for ind in range(np.shape(dict_mapping["Mapping"])[1])])] = 2
    # Get output array
    no_mutations_type = np.zeros(np.shape(dict_mapping["Mapping"])[1])
    # Search for residues with no mutations (AA type)
    no_mutations_type[np.where([True if (all(dict_mapping_colors["Mapping Amino Acid Type reduced"][1:,ind]>=7)) and not (all(dict_mapping_colors["Mapping Amino Acid Type reduced"][1:,ind]==7)) else False for ind in range(np.shape(dict_mapping["Mapping"])[1])])] = 1
    # Remove overall non-mapping residues
    no_mutations_type[np.where([True if (all(dict_mapping_colors["Mapping Amino Acid Type reduced"][1:,ind]==7)) else False for ind in range(np.shape(dict_mapping["Mapping"])[1])])] = 2    
    # Generate colormap
    cmap_bin = ListedColormap([dict_colors["No Mutations"][ind] for ind in [0,1,2]])
    # Update dictionary
    dict_mapping_colors.update({"No Mutations":no_mutations,
                                "No Type Mutations":no_mutations_type,
                                "Colormap No Mutations":cmap_bin,
                                "Colormap No Type Mutations":cmap_bin})
    
    # If ligands is present get mapping for binding site
    if ligands_exist:
        binding_site = np.zeros(np.shape(dict_mapping["Mapping"])[1])
        # Check whether atoms located in reach of any ligand
        try:
            # get binding site (indexing in pdb starts with 1 => -1)
            binding_site[dict_molecules[reference]["Ligand Indices"]-1]=1
        except:
            ligands_exist = False
        cmap_bin = ListedColormap([dict_colors["Binding Site"][ind] for ind in [0,1]])
        dict_mapping_colors.update({"Binding Site":binding_site,
                                    "Colormap Binding Site":cmap_bin})
    
    return dict_mapping_colors, ligands_exist

def plot_Feature_comparison(key, dict_mapping, dict_mapping_colors, dict_boundaries, output_path=".", ligands_exist=True, do_plots_binding_site=False, dict_molecules=None, reference=None, fontsize=10, keys_ordered = None):
    """
    Plots a comparison for selected features.

    Parameters
    ----------
    key : str
        Feature to be plotted: ["Amino Acid","Amino Acid reduced","Amino Acid Type","Amino Acid Type reduced","DSSP","DSSP reduced","SASA","pLDDT"].
    dict_mapping : dict
        Dictionary containg all mapping information with respect to the reference structure.
    dict_mapping_colors : dict
        Dictionary containg colors for mapping.
    dict_boundaries : dict
        Boundaries for every key. Suffix "reduced" introduces a color for consistency between all structures.
    output_path : str, optional
        Path for data ouput. The default is ".".
    ligands_exist : bool, optional
        True if ligand exist. The default is True.
    do_plots_binding_site : bool, optional
        Plot only the binding site instead of all residues. The default is False.
    dict_molecules : dict
        Dictonary containing structural properties.
    reference : str
        Reference file.
    fontsize : int, optional
        Font size for plot. The default is 10.
    keys_ordered : list of str, optional
        Ordering of keys if needed. The default is None.

    Returns
    -------
    None.

    """
    # Check whether input for binding site is provided
    if do_plots_binding_site and not (dict_molecules and reference):
        raise ValueError("dict_molecule and reference needed for plotting of binding site only.")
    # set width for plotting ratio
    if do_plots_binding_site:
        size_full = 100
    else:
        size_full = 20
        
    if keys_ordered:
        order = [dict_mapping["Keys"].index(key) for key in keys_ordered]
    else:
        order = list(range(len(dict_mapping["Keys"])))
    
    # Check whether key exist
    if key not in ["Amino Acid","Amino Acid reduced","Amino Acid Type","Amino Acid Type reduced","DSSP","DSSP reduced","SASA","pLDDT"]:
        raise ValueError("Feature not known")
    
    # Setup figure
    fig = plt.figure()
    fig.set_size_inches(20,6)
    # Set up layout w.r.t. binding site is known/exists
    if ligands_exist:
        spec = fig.add_gridspec(ncols=1, nrows=9, height_ratios=[len(dict_mapping["Mapping"]),1,1,1,1,1,1,1,1], hspace=0)
        axl = fig.add_subplot(spec[7, 0])
        ax6 = fig.add_subplot(spec[8, 0])
    else:
        spec = fig.add_gridspec(ncols=1, nrows=8, height_ratios=[len(dict_mapping["Mapping"]),1,1,1,1,1,1,1], hspace=0)
        ax6 = fig.add_subplot(spec[7, 0])
    # Generate additional (summary) axes
    ax = fig.add_subplot(spec[0, 0])
    ax1 = fig.add_subplot(spec[2, 0])
    ax2 = fig.add_subplot(spec[3, 0])
    ax3 = fig.add_subplot(spec[4, 0])
    ax4 = fig.add_subplot(spec[5, 0])
    ax5 = fig.add_subplot(spec[6, 0])
    
    # Plot main figure
    if key in ["Amino Acid","Amino Acid reduced","Amino Acid Type","Amino Acid Type reduced","DSSP","DSSP reduced"]:
        if do_plots_binding_site:
            im = ax.matshow(dict_mapping_colors["Mapping "+key][order][:,dict_molecules[reference]["Ligand Indices"]-1], cmap = dict_mapping_colors["Colormap "+key], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key][0], vmax=dict_boundaries[key][1])
        else:
            im = ax.matshow(dict_mapping_colors["Mapping "+key][order], cmap = dict_mapping_colors["Colormap "+key], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key][0], vmax=dict_boundaries[key][1])
    else:
        if do_plots_binding_site:
            im = ax.matshow(dict_mapping[key][order][:,dict_molecules[reference]["Ligand Indices"]-1], cmap = dict_mapping_colors["Colormap "+key], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key][0], vmax=dict_boundaries[key][1])            
        else:
            im = ax.matshow(dict_mapping[key][order], cmap = dict_mapping_colors["Colormap "+key], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key][0], vmax=dict_boundaries[key][1])
    # Layout main figure
    ax.set_yticks(np.arange(len(dict_mapping["Keys"])))
    ax.set_yticklabels([dict_mapping["Keys"][label] for label in order],fontsize=fontsize)
    if do_plots_binding_site:
        ax.set_xticks(np.arange(10,np.shape(dict_molecules[reference]["Ligand Indices"])[0],10))        
    else:
        ax.set_xticks(np.arange(50,np.shape(dict_mapping["Mapping"])[1],50))
    # Plot subfigures including layout
    key_s = "pLDDT"
    if do_plots_binding_site:
        ax1.matshow(dict_mapping[key_s][0,dict_molecules[reference]["Ligand Indices"]-1].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])        
    else:
        ax1.matshow(dict_mapping[key_s][0,:].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
    ax1.set_yticks([0])
    ax1.set_yticklabels(["pLDDT"])
    ax1.set_xticks([])
    
    key_s = "DSSP"
    if do_plots_binding_site:
        ax2.matshow(dict_mapping_colors["Mapping "+key_s][0,dict_molecules[reference]["Ligand Indices"]-1].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])        
    else:
        ax2.matshow(dict_mapping_colors["Mapping "+key_s][0,:].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
    ax2.set_yticks([0])
    ax2.set_yticklabels(["DSSP"])
    ax2.set_xticks([])

    key_s = "SASA"
    if do_plots_binding_site:
        ax3.matshow(dict_mapping[key_s][0,dict_molecules[reference]["Ligand Indices"]-1].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])     
    else:
        ax3.matshow(dict_mapping[key_s][0,:].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
    ax3.set_yticks([0])
    ax3.set_yticklabels(["SASA"])
    ax3.set_xticks([])
    
    if ligands_exist:
        key_s = "Binding Site"
        if do_plots_binding_site:
            axl.matshow(dict_mapping_colors[key_s][dict_molecules[reference]["Ligand Indices"]-1].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
        else:
            axl.matshow(dict_mapping_colors[key_s].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
        axl.set_yticks([0])
        axl.set_yticklabels(["Binding Site"])
        axl.set_xticks([])

    key_s = "No Mutations"
    if do_plots_binding_site:
        ax4.matshow(dict_mapping_colors[key_s][dict_molecules[reference]["Ligand Indices"]-1].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
    else:
        ax4.matshow(dict_mapping_colors[key_s].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
    ax4.set_yticks([0])
    ax4.set_yticklabels(["No Mutations"])
    ax4.set_xticks([])
    
    key_s = "No Type Mutations"
    if do_plots_binding_site:
        ax5.matshow(dict_mapping_colors[key_s][dict_molecules[reference]["Ligand Indices"]-1].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
    else:
        ax5.matshow(dict_mapping_colors[key_s].reshape(1,-1), cmap = dict_mapping_colors["Colormap "+key_s], aspect = np.shape(dict_mapping["Mapping"])[1]/size_full, vmin=dict_boundaries[key_s][0], vmax=dict_boundaries[key_s][1])
    ax5.set_yticks([0])
    ax5.set_yticklabels(["No Type Mutations"])
    ax5.set_xticks([])
    
    key_s = "Amino Acid Type"
    if do_plots_binding_site:
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
        cbar.set_label(key,rotation=90, fontsize=fontsize)
    
    if do_plots_binding_site:
        plt.savefig(os.path.join(output_path,"FIG_BS_"+key.replace(" ","_")+".png"), bbox_inches="tight")
    else:
        plt.savefig(os.path.join(output_path,"FIG_"+key.replace(" ","_")+".png"), bbox_inches="tight")
    plt.close()

def write_output(dict_molecules, dict_mapping, dict_mapping_colors, reference, output_path_data="."):
    """
    Saves all dictionaries to file.

    Parameters
    ----------
    dict_molecules : dict
        Dictonary containing structural properties.
    dict_mapping : dict
        Dictionary containg all mapping information with respect to the reference structure.
    dict_mapping_colors : dict
        Dictionary containg colors for mapping.
    reference : str
        Reference file.
    output_path_data : str, optional
        Output path for the data written to dsic. The default is ".".

    Returns
    -------
    None.

    """
    # Merge all dictionaries into one
    dict_full = {"Molecules":dict_molecules,
                 "Mapping":dict_mapping,
                 "Mapping Colors":dict_mapping_colors}
    # Save dictonary to file
    pickle.dump(dict_full, open(os.path.join(output_path_data,reference+".pkl"),"wb"))

def get_clustered_network(dict_molecules, dict_mapping_colors, reference, key = "No Mutations", dist_cut=10, seed=42, output_path=".", write_vmd=False, output_path_vmd=".", file_name_tcl="out_vmd.tcl", vmd_out="Structure", files_proteins=None, summarize=False):
    """
    Constructs and clusters the residues network for conserved amino acids.

    Parameters
    ----------
    dict_molecules : dict
        Dictonary containing structural properties.
    dict_mapping : dict
        Dictionary containg all mapping information with respect to the reference structure.
    reference : str
        Reference file.
    key : str, optional
        Selection of residues to be clustered:  ["No Mutations","No Type Mutations","Binding Site"]. The default is "No Mutations".
    dist_cut : float, optional
        Distance cutoff for network construction. The default is 10.
    seed : int, optional
        Seed used for clustering. The default is 42.
    output_path : str, optional
        Output path for the data written to dsic. The default is ".".
    write_vmd : bool, optional
        True if VMD input should be written. The default is False.
    output_path_vmd : str, optional
        Output path for the VMD input. The default is ".".
    file_name_tcl : str, optional
        Name of the file for the VMD input. The default is "out_vmd.tcl".
    vmd_out : str, optional
        Name of the VMD output. The default is "Structure".
    files_proteins : list of str
        List of all protein structures.
    summarize : bool, optional
        Summarization of the network. The default is False.

    Returns
    -------
    labels_mol : list of int
        Assignment of residues to clusters. 0 denotes residues not used for network construction.

    """
    if key not in ["No Mutations","No Type Mutations","Binding Site"]:
        raise ValueError("key not found.")

    # Extract selection
    selection = dict_mapping_colors[key]

    # Extract CA-coordinates for reference
    c_alpha = dict_molecules[reference]["C_alpha"]

    # remove all with mutations
    indices = [ind for ind in range(len(c_alpha)) if selection[ind]==1]
    c_alpha_red = [c_alpha[ind] for ind in indices]

    # Set up graph for clustering
    G = get_Graph_network(indices, coordinates=c_alpha_red, dist_cut=dist_cut)

    # Leiden clustering
    cluster_list, labels = cluster_Network(G, seed=seed)
    
    # Plot graph
    plot_clustered_network(G, labels, output_path=output_path, summarize=summarize, dist_cut=dist_cut, coordinates=c_alpha_red)

    # Get indices for molecules
    indices_mol = [[ind for ind in range(len(c_alpha)) if selection[ind]!=1]]+[[indices[i] for i in item] for item in cluster_list]
    labels_mol = np.zeros(len(c_alpha))
    for ind,cluster in enumerate(indices_mol):
        labels_mol[cluster] = ind

    # Write .tcl
    if write_vmd and files_proteins:
        # Get file name for reference structure
        file_name = files_proteins[0]
        # Write VMD file to disc
        write_vmd_output(dict_molecules, reference, labels_mol, file_name, output_path_vmd=output_path_vmd, file_name_tcl=file_name_tcl, vmd_out=vmd_out)
    return labels_mol
    
def get_Graph_network(indices, coordinates, dist_cut=10):
    """
    Generates the network graph.

    Parameters
    ----------
    indices : list of int
        Indices for residue selection.
    coordinates : list of arr
        Positions of selected atoms.
    dist_cut : float, optional
        Distance cutoff for network construction. The default is 10.

    Returns
    -------
    G : obj
        Network graph generated with igraph.

    """
    G = ig.Graph()
    G.add_vertices(len(indices))

    # Calculate distances (Tree)
    tree = scipy.spatial.cKDTree(coordinates)
    neighbors = []
    # Get neighbors
    for atom_coord in coordinates:
        neighbors.append(tree.query_ball_point(atom_coord,dist_cut))
    # Get edges
    # j>i Removes duplicates and diagonal Elements
    edges = [(i,j) for i in range(len(neighbors)) for j in neighbors[i] if j>i]
    # Add edges based on cutoff
    G.add_edges(edges)
    return G

def cluster_Network(G, seed=42):
    """
    Clusters network using the Leiden algorithm.

    Parameters
    ----------
    G : obj
        Network graph generated with igraph.
    seed : int, optional
        Seed used for clustering. The default is 42.
        
    Returns
    -------
    cluster_list : list of arr
        list of arrays containg indices of residues belonging to the respective cluster.
    labels : list of int
        Assignment of residues to clusters for atoms used in network construction.

    """
    # Get partition
    partition = la.find_partition(G, la.CPMVertexPartition, seed=seed, resolution_parameter=0.05)

    # Extract labels
    labels = partition.membership
    # Convert labels to list
    cluster_list = [np.where(labels==label)[0] for label in np.unique(labels)]  
    return cluster_list, labels

def plot_clustered_network(G, labels, output_path=".", summarize=False, dist_cut=10, coordinates=None):
    """
    Plot the clustered network.

    Parameters
    ----------
    G : obj
        Network graph generated with igraph.
    labels : list of int
        Assignment of residues to clusters for atoms used in network construction.
    output_path : str, optional
        Path for data ouput. The default is ".".
    summarize : bool, optional
        Summarization of the network. The default is False.
    dist_cut : float, optional
        Distance cutoff for network construction. The default is 10.
    coordinates : list of arr
        Positions of selected atoms. Needed if summarized is True. The default is None.

    Returns
    -------
    None.

    """
    # Get colormap
    cmap = mpl.cm.get_cmap('jet_r')
    # Get discrete colors
    colors_to_choose = np.linspace(0+1/len(set(labels)),1,len(set(labels)))
    colors = [cmap(el) for el in colors_to_choose]
    
    # Plot
    fig, ax = plt.subplots()
    ig.plot(G, layout=G.layout_kamada_kawai(), target=ax, vertex_color = [colors[cluster] for cluster in labels])
    plt.axis("off")
    plt.savefig(os.path.join(output_path,"FIG_clustered_network.png"), bbox_inches="tight")
    
    if summarize:
        if not coordinates:
            raise ValueError("Coordnates have to be provided.")
        else:
            dict_weights = {}
            for i in np.unique(labels)[:-1]:
                for j in np.unique(labels)[i+1:]:
                    tree = scipy.spatial.cKDTree([coordinates[ind] for ind in [index for index,label in enumerate(labels) if label==i]])
                    neighbors = []
                    for atom_coord in [coordinates[ind] for ind in [index for index,label in enumerate(labels) if label==j]]:
                        neighbors.append(tree.query_ball_point(atom_coord,dist_cut))
                    dict_weights.update({str(i)+"_"+str(j):len([el for item in neighbors for el in item])})
            
            G_sum = ig.Graph()
            G_sum.add_vertices(len(set(labels)))
    
            edges = [(i,j) for i in np.unique(labels)[:-1] for j in np.unique(labels)[i+1:] if dict_weights[str(i)+"_"+str(j)]>0]
            try:
                weights = [dict_weights[str(i)+"_"+str(j)] for i in np.unique(labels)[:-1] for j in np.unique(labels)[i+1:] if dict_weights[str(i)+"_"+str(j)]>0]
                weights = (weights/np.max(weights)*9)+1
            except:
                weights = []
            size = [len(np.where(labels==label)[0]) for label in np.unique(labels)]
            size = (size/np.max(size)*10)+10
    
            G_sum.add_edges(edges)
            ig.plot(G_sum, os.path.join(output_path,"FIG_clustered_network_summarized.png"), layout="kk", vertex_color = [colors[cluster] for cluster in np.unique(labels)], edge_width=weights, vertex_size = size)
    
def write_vmd_output(dict_molecules, reference, labels_mol, file_name, output_path_vmd=".", file_name_tcl="out_vmd.tcl", vmd_out="Structure"):
    """
    Writes VMD output for automated structure view generation.

    Parameters
    ----------
    dict_molecules : dict
        Dictonary containing structural properties.
    reference : str
        Reference file.
    labels_mol : list of int
        Assignment of residues to clusters. 0 denotes residues not used for network construction.
    file_name : str
        File name for the reference structure.
    output_path_vmd : str, optional
        Output path for the VMD input. The default is ".".
    file_name_tcl : str, optional
        Name of the file for the VMD input. The default is
    vmd_out : str, optional
        Name of the VMD output. The default is "Structure".

    Returns
    -------
    None.

    """
    with open (os.path.join(output_path_vmd,file_name_tcl),"w") as file:
        file.write("# Use for GUI: vmd -e %s\n"%file_name_tcl)
        file.write("# Use for pipeline w/o GUI: vmd -dispdev text -eofexit < %s\n"%file_name_tcl)
        file.write("\n")
        file.write("display backgroundgradient off\naxes location off\ndisplay depthcue off\ndisplay update on")
        file.write("color add item Display Background black\ncolor Display Background black\ncolor scale method RGB\ncolor scale midpoint 0.50\ncolor scale min 0.00\n")
        file.write("mol default material AOEdgy\nmol default style NewCartoon\nmol default selection {all}\nmol default color Beta\n")
        file.write("\n")
        file.write("mol new "+file_name.replace(" ","\ ")+"\n")
        file.write("set full [atomselect top protein]\n")
        file.write("$full set beta {")
        for index in dict_molecules[reference]["Indices"]:
            file.write("%d "%labels_mol[index-1])
        file.write("}\n")
        file.write("\n")
        file.write("set color_start [colorinfo num]\ncolor change RGB [expr $color_start +0] 1 1 1\ndisplay resize\n")
        file.write("\n")
        file.write("set outname %s\n"%os.path.join(output_path_vmd,vmd_out))
        file.write("render TachyonInternal ${outname}_Z.tga\nrotate x by 90 degrees\nrender TachyonInternal ${outname}_Y.tga\nrotate y by 270 degrees\nrender TachyonInternal ${outname}_X.tga\n")
        file.write("\n")
        file.write("# For high resolution use in TK console:\n# render TachyonInternal ${outname}_high_res.tga")

def write_Structural_MSA(dict_mapping, dict_amino_acids,  output_path=".", output_name="Structural_MSA.FASTA"):
    """
    Generates FASTA output for structural alignment.

    Parameters
    ----------
    dict_mapping : dict
        Dictionary containg all mapping information with respect to the reference structure.
    dict_amino_acids : dict
        3-Letter to 1-Letter amino acid translation.
    output_path : str, optional
        Output path for the data written to dsic. The default is ".".
    output_name : str, optional
        Name of the output. The default is "Structural_MSA.FASTA".

    Returns
    -------
    None.

    """
    with open(os.path.join(output_path,output_name),"w") as file:
        for ind,key in enumerate(dict_mapping["Keys"]):
            file.write(">"+key+"\n")
            for item in dict_mapping["Amino Acid"][ind,:]:
                file.write(dict_amino_acids[item])
            file.write("\n")