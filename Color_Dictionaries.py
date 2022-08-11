#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Oliver Lemke
"""

##################
## Dictionaries ##
##################

def get_dictionaries():
    """
    Get dictionaries for structure mapping. Amino acid type assignment, colors/colormaps and limits for plotting.

    Returns
    -------
    dict_atom_type : dict
        Assignment of each amino acid to a property.
    dict_colors : dict
        Colors for amino acids, amino acid types, DSSP, Binding site, Mutation sites. Colormaps for SASA and pLDDT.
    dict_boundaries : dict
        Boundaries for every key. Suffix "reduced" introduces a color for consistency between all structures.
    dict_amino_acids : dict
        3-Letter to 1-Letter amino acid translation.
    """
    
    ## Amino Acid Type
    dict_atom_type = {"LYS":"Basic","ARG":"Basic","HIS":"Basic",                                    # K,R,H -> Basic
                      "GLU":"Acidic","ASP":"Acidic",                                                # E,D -> Acidic
                      "TRP":"Aromatic","PHE":"Aromatic","TYR":"Aromatic",                           # W,F,Y -> Aromatic
                      "THR":"Polar","SER":"Polar","CYS":"Polar","GLN":"Polar","ASN":"Polar",        # T,S,C,Q,N -> Polar
                      "ALA":"Apolar","VAL":"Apolar","LEU":"Apolar","ILE":"Apolar","MET":"Apolar",   # A,V,L,I,M -> Apolar
                      "PRO":"Break",                                                                # P -> Secondary Structure Break
                      "GLY":"Flexible",                                                             # G -> Higher flexibility
                      "XYZ":"NoMap"}
    
    ## max. SASA
    dict_sasa_max = {'ALA': 1.2831596, 'CYS': 1.6075546, 'ASP': 1.6942018, 'GLU': 1.9520556, 'PHE': 2.3931725, 
                     'GLY': 0.9859995, 'HIS': 2.1476686, 'ILE': 1.9781175, 'LYS': 2.4149895, 'LEU': 1.9980704, 
                     'MET': 2.1391654, 'ASN': 1.8180864, 'PRO': 1.6501249, 'GLN': 1.996297, 'ARG': 2.7603362, 
                     'SER': 1.4527341, 'THR': 1.6533911, 'VAL': 1.7248302, 'TRP': 2.773364, 'TYR': 2.5169885,
                     'XYZ': 1.}
    
    ## Colors
    dict_colors = {"Amino Acid":
                       {"LYS":"#154360","ARG":"#21618C","HIS":"#5499C7",                                    # Blue
                        "GLU":"#641E16","ASP":"#CB4335",                                                    # Red
                        "TRP":"#145A32","PHE":"#239B56","TYR":"#52BE80",                                    # Green
                        "THR":"#4A235A","SER":"#6C3483","CYS":"#9B59B6","GLN":"#C39BD3","ASN":"#D2B4DE",    # Violet
                        "ALA":"#B9770E","VAL":"#D68910","LEU":"#F4D03F","ILE":"#F1C40F","MET":"#D4AC0D",    # Yellow/Orange
                        "PRO":"#1C2833",                                                                    # Black
                        "GLY":"#F0F3F4",                                                                    # Light Gray
                        "XYZ":"#D0D0D0"},                                                                   # Gray
                   "Amino Acid Type":
                       {"Basic":"#3498DB",        # Blue (N)
                        "Acidic":"#C0392B",       # Red (O)   
                        "Aromatic":"#28B463",     # Green
                        "Polar":"#8E44AD",        # Violet
                        "Apolar":"#E67E22",       # Orange
                        "Break":"#212F3D",        # Dark
                        "Flexible":"#F7DC6F",     # Yellow
                        "NoMap":"#D0D0D0"},       # Gray
                   "DSSP":
                       {"E":"C3",                 # Red
                        "H":"C0",                 # Blue
                        "C":"C8",                 # Yellow
                        "NoMap":"#D0D0D0"},       # Gray
                   "Binding Site":
                       {0:"w",                    # White (Background)
                        1:"k"},                   # Black (Highlight)
                   "No Mutations":
                       {0:"w",                    # White (Background)
                        1:"k",                    # Black (Highlight)
                        2:"#D0D0D0"},             # Gray
                   "Characteristic":
                        {"pLDDT":"RdPu",
                         "SASA":"Blues"}}
    
    # Boundaries for heatmap plots
    dict_boundaries = {"Amino Acid":[-0.5, 20.5],               # 20 Canonical Amino Acids + NoMap (XYZ)
                       "Amino Acid reduced":[-0.5, 21.5],       # 1 Additional: Match
                       "Amino Acid Type":[-0.5, 7.5],           # 7 Amino Acid Types + NoMap
                       "Amino Acid Type reduced":[-0.5, 8.5],   # 1 Additional: Match
                       "DSSP":[-0.5,3.5],                       # 3 simplified types
                       "DSSP reduced":[-0.5,4.5],               # 1 Additional: Match
                       "SASA":[0,3],                            # Scale (Min,Max) (estimate)
                       "pLDDT":[0,100],                         # Scale (Min,Max)
                       "Binding Site":[0,1],                    # Present or absent
                       "No Mutations":[0,2],                    # Present, absent or NoMap
                       "No Type Mutations":[0,2]}               # Present, absent or NoMap
                       
    dict_amino_acids = {"ALA":"A",
                        "CYS":"C",
                        "ASP":"D",
                        "GLU":"E",
                        "PHE":"F",
                        "GLY":"G",
                        "HIS":"H",
                        "ILE":"I",
                        "LYS":"K",
                        "LEU":"L",
                        "MET":"M",
                        "ASN":"N",
                        "PRO":"P",
                        "GLN":"Q",
                        "ARG":"R",
                        "SER":"S",
                        "THR":"T",
                        "VAL":"V",
                        "TRP":"W",
                        "TYR":"Y",
                        "XYZ":"-"}
    
    return dict_atom_type, dict_colors, dict_boundaries, dict_amino_acids, dict_sasa_max
    

