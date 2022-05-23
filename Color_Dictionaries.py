#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:46:47 2022

@author: Oliver
"""

##################
## Dictionaries ##
##################

def get_dictionaries():
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
                        1:"k",
                        2:"#D0D0D0"},
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
                       "No Mutations":[0,2],
                       "No Type Mutations":[0,2]}
    return dict_atom_type, dict_colors, dict_boundaries
    

