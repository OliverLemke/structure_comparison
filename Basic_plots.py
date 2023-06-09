#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:38:32 2023

@author: oliverlemke
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr, kendalltau
from matplotlib.offsetbox import AnchoredText
import matplotlib as mpl
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_curve, roc_auc_score
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit

#%%
# TODO
## Error handling for missing keys

#%%

# Histogram
def plot_hist(data_frame, keys, outfile="Out_hist.pdf", x_label="x", y_label="y", fs=15, fs_legend=15, n_bins=20, smoothing_factor=1e-10, legend_loc="upper left", x_lim=None, grid=True):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)

    max_y = 0
    
    if not x_lim:
        x_lim = (np.nanmin(data_frame[keys.keys()].values)-(0.05*np.nanmax(data_frame[keys.keys()].values)),np.nanmax(data_frame[keys.keys()].values)+(0.05*np.nanmax(data_frame[keys.keys()].values)))
    
    for key in keys:

        hist = np.histogram(data_frame[key].values, range=x_lim, bins=n_bins, density=True)
        x = (hist[1][1:]+hist[1][:-1])/2

        spl = UnivariateSpline(np.insert(x,len(x),x_lim[1]),np.insert(hist[0],len(hist[0]),hist[0][-1]))
        spl.set_smoothing_factor(smoothing_factor)
        
        xs = np.linspace(x_lim[0],x_lim[1],1000)
        
        ax.plot(xs, spl(xs), color=keys[key]["Color"], label=keys[key]["Label"])
        #ax.plot(x, hist[0], color=keys[key]["Color"])
        ax.fill_between(xs, np.zeros(len(xs)), spl(xs), color=keys[key]["Color"], alpha=.6)
        
        max_y = np.nanmax((max_y,np.nanmax(hist[0])))
    
    if grid:
        ax.grid(axis='both', color='0.8')
           
    ax.set_xlim(x_lim)           
    ax.set_ylim(0,max_y*1.1)

    ax.set_xlabel(x_label, fontsize=fs)
    ax.set_ylabel(y_label, fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs)
    
    try:
        legend = plt.legend(loc=legend_loc, fontsize=fs_legend, shadow=True, fancybox=True, framealpha=1)
    except:
        print("legend_loc not found. Using upper left as a default.")
        legend = plt.legend(loc="upper left", fontsize=fs_legend, shadow=True, fancybox=True, framealpha=1)
    legend.get_frame().set_linewidth(2)

    for legobj in legend.legendHandles:
        legobj.set_linewidth(5)

    plt.tight_layout()
    plt.savefig(outfile)
    
def plot_hist_selection(data_frame, selections, ref_key, outfile="Out_hist_selection.pdf", x_label="x", y_label="y", fs=15, fs_legend=15, n_bins=20, smoothing_factor=1e-10, legend_loc="upper left", x_lim=None, grid=True, plot_full=True):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)

    max_y = 0
    key_x = list(ref_key.keys())[0]
    
    if not x_lim:
        x_lim = (np.nanmin(data_frame[key_x].values)-(0.05*np.nanmax(data_frame[key_x].values)),np.nanmax(data_frame[key_x].values)+(0.05*np.nanmax(data_frame[key_x].values)))
    
    for selection in selections:

        hist = np.histogram(data_frame.loc[selections[selection]["Indices"],key_x].values, range=x_lim, bins=n_bins, density=True)
        x = (hist[1][1:]+hist[1][:-1])/2

        spl = UnivariateSpline(np.insert(x,len(x),x_lim[1]),np.insert(hist[0],len(hist[0]),hist[0][-1]))
        spl.set_smoothing_factor(smoothing_factor)
        
        xs = np.linspace(x_lim[0],x_lim[1],1000)
        
        ax.plot(xs, spl(xs), color=selections[selection]["Color"], label=selections[selection]["Label"])
        ax.fill_between(xs, np.zeros(len(xs)), spl(xs), color=selections[selection]["Color"], alpha=.6)
        
        max_y = np.nanmax((max_y,np.nanmax(hist[0])))
    
    if plot_full:
        hist = np.histogram(data_frame[key_x].values, range=x_lim, bins=n_bins, density=True)
        x = (hist[1][1:]+hist[1][:-1])/2

        spl = UnivariateSpline(np.insert(x,len(x),x_lim[1]),np.insert(hist[0],len(hist[0]),hist[0][-1]))
        spl.set_smoothing_factor(smoothing_factor)
        
        xs = np.linspace(x_lim[0],x_lim[1],1000)
        
        ax.plot(xs, spl(xs), color=ref_key[key_x]["Color"], label=ref_key[key_x]["Label"], alpha=.6)
        ax.fill_between(xs, np.zeros(len(xs)), spl(xs), color=ref_key[key_x]["Color"], alpha=.4)
        
        max_y = np.nanmax((max_y,np.nanmax(hist[0])))
    
    if grid:
        ax.grid(axis='both', color='0.8')
           
    ax.set_xlim(x_lim)           
    ax.set_ylim(0,max_y*1.1)

    ax.set_xlabel(x_label, fontsize=fs)
    ax.set_ylabel(y_label, fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs)
    
    try:
        legend = plt.legend(loc=legend_loc, fontsize=fs_legend, shadow=True, fancybox=True, framealpha=1)
    except:
        print("legend_loc not found. Using upper left as a default.")
        legend = plt.legend(loc="upper left", fontsize=fs_legend, shadow=True, fancybox=True, framealpha=1)
    legend.get_frame().set_linewidth(2)

    for legobj in legend.legendHandles:
        legobj.set_linewidth(5)

    plt.tight_layout()
    plt.savefig(outfile)
    
def fit_1_over_x(x,a,b,c):
    return a/(x+b)+c

# Scatter    
def plot_correlation_scatter(data_frame, keys, outfile="Out_correlation_scatter.pdf", fs=20, fs_text=15, n_bins=20, smoothing_factor=1e-10, text_loc="lower right", color = "C0", pearson = True, spearman = True, kendall = False, p_pearson = None, p_spearman = None, p_kendall = None, x_lim = None, y_lim = None, plot_linreg = True, plot_xy = False, grid = True, highlight = None, legend_loc = "upper left", plot_1_over_x=False, highlight_size=150):

    # Formatter for same precision labels
    # Add title
    # Add gradient for highlighting
    ## Use if gradient before if highlight, use cmap -> Include colorbar!!!!
    ### Use scatter(x,y,c=value,cmap)
    
    fig = plt.figure()
    fig.set_size_inches(7.5,7.5)
    gs = gridspec.GridSpec(2,2,width_ratios=[10,2],height_ratios=[2,10], wspace=0.01, hspace=0.01)
    
    key_x = list(keys.keys())[0]
    key_y = list(keys.keys())[1]
    
    data_to_plot = data_frame.copy()[[key_x,key_y]]
    data_to_plot.dropna(inplace=True)
    
    ax_scatter = fig.add_subplot(gs[1,0])
    ax_scatter.scatter(data_to_plot[key_x],data_to_plot[key_y],facecolors="None",edgecolors=color, marker=".",s=40)#,alpha=.6)
    
    if highlight:
        for key in highlight:
            ax_scatter.scatter(data_to_plot.loc[highlight[key]["Indices"],key_x],data_to_plot.loc[highlight[key]["Indices"],key_y],facecolors=highlight[key]["Color"],edgecolors=highlight[key]["Color"], marker=".",s=highlight_size, label=key)#,alpha=.6)

    if pearson:
        pearson_corr = pearsonr(data_to_plot[key_x],data_to_plot[key_y])
    else:
        pearson_corr = None
    if spearman:
        spearman_corr = spearmanr(data_to_plot[key_x],data_to_plot[key_y])
    else:
        spearman_corr = None
    if kendall:
        kendall_corr = kendalltau(data_to_plot[key_x],data_to_plot[key_y])
    else:
        kendall_corr = None
    
    ax_scatter.set_xlabel(keys[key_x]["Label"], fontsize=fs)
    ax_scatter.set_ylabel(keys[key_y]["Label"], fontsize=fs)
    ax_scatter.tick_params(axis="both", labelsize=fs)
    
    if not x_lim:
        x_lim = (np.nanmin(data_to_plot[key_x])-(0.05*np.nanmax(data_to_plot[key_x])),np.nanmax(data_to_plot[key_x])+(0.05*np.nanmax(data_to_plot[key_x])))
    ax_scatter.set_xlim(x_lim)
        
    if not y_lim:
        y_lim = (np.nanmin(data_to_plot[key_y])-(0.05*np.nanmax(data_to_plot[key_y])),np.nanmax(data_to_plot[key_y])+(0.05*np.nanmax(data_to_plot[key_y])))
    ax_scatter.set_ylim(y_lim)    
        
    if plot_linreg:
        coef = np.polyfit(data_to_plot[key_x],data_to_plot[key_y],1)
        poly1d_fn = np.poly1d(coef)
        ax_scatter.plot(x_lim,poly1d_fn(x_lim),c="k")
    
    if plot_1_over_x:
        popt, pcov = curve_fit(fit_1_over_x, data_to_plot[key_x], data_to_plot[key_y])
        xs = np.linspace(x_lim[0], x_lim[1],100)
        ax_scatter.plot(xs,fit_1_over_x(xs, *popt),c="k")
        
    if plot_xy:
        ax_scatter.plot(x_lim, x_lim, ls=":", c="k")
        
    if grid:
        ax_scatter.grid(axis='both', color='0.8')
        
    if pearson or spearman or kendall:
        text = ""
        if pearson:
            if p_pearson:
                text += "R_Pearson = {0:.2f}\np_Pearson = {1:.2e}".format(pearson_corr[0],p_pearson)            
            else:
                text += "R_Pearson = {0:.2f}\np_Pearson = {1:.2e}".format(pearson_corr[0],pearson_corr[1])
        if pearson and spearman:
            text +="\n"
        if spearman:
            if p_spearman:
                text += "R_Spearman = {0:.2f}\np_Spearman = {1:.2e}".format(spearman_corr[0],p_spearman)
            else:
                text += "R_Spearman = {0:.2f}\np_Spearman = {1:.2e}".format(spearman_corr[0],spearman_corr[1])
        if (kendall and spearman) or (kendall and pearson):
            text +="\n"
            if p_kendall:
                text += "Kendall_tau = {0:.2f}\np_Kendall = {1:.2e}".format(kendall_corr[0],p_kendall)
            else:
                text += "Kendall_tau = {0:.2f}\np_Kendall = {1:.2e}".format(kendall_corr[0],kendall_corr[1])
        try:    
            anchored_text = AnchoredText(text, loc=text_loc, prop=dict(size=fs_text))
        except:
            print("text_loc not found. Using lower right as a default.")
            anchored_text = AnchoredText(text, loc="lower right", prop=dict(size=fs_text))
            #ax_scatter.add_artist(AnchoredText(text, loc="lower right"))
    
        anchored_text.patch.set_alpha(0.5)
        anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax_scatter.add_artist(anchored_text)
    
    if highlight:
        try:
            ax_scatter.legend(loc=legend_loc, fontsize=fs_text)
        except:
            print("legend_loc not found. Using upper left as a default.")
            ax_scatter.legend(loc="upper_left", fontsize=fs_text)
        
    #
    ax_hist_x = fig.add_subplot(gs[0,0])
    ax_hist_x.axis("off")
    
    hist = np.histogram(data_to_plot[key_x].values, range=x_lim, bins=n_bins, density=True)
    x = (hist[1][1:]+hist[1][:-1])/2
    
    spl = UnivariateSpline(np.insert(x,len(x),hist[1][-1]),np.insert(hist[0],len(hist[0]),hist[0][-1]))
    spl.set_smoothing_factor(smoothing_factor)
    
    xs = np.linspace(x_lim[0],x_lim[1],1000)
    
    ax_hist_x.plot(xs, spl(xs), color=keys[key_x]["Color"])
    ax_hist_x.fill_between(xs, np.zeros(len(xs)), spl(xs), color=keys[key_x]["Color"], alpha=.6)
    ax_hist_x.plot([np.nanmedian(data_to_plot[key_x]),np.nanmedian(data_to_plot[key_x])],[0,np.nanmax(hist[0])*1.1],c="k",ls="--")
    
    ax_hist_x.set_xlim(x_lim)           
    ax_hist_x.set_ylim(0,np.nanmax(hist[0])*1.1)
    
    #if highlight_hist and highlight:
    #    for key in highlight:
    #        hist_high_n = np.histogram(data_to_plot.loc[highlight[key]["Indices"],key_x].values, range=x_lim, bins=n_bins, density=True)
    #        hist_high = (hist_high_n[0]/np.max(hist_high_n[0])*np.max(hist[0]),hist_high_n[1])
    #        spl = UnivariateSpline(np.insert(x,len(x),hist_high[1][-1]),np.insert(hist_high[0],len(hist_high[0]),hist_high[0][-1]))
    #        spl.set_smoothing_factor(smoothing_factor)
    #        ax_hist_x.plot(xs, spl(xs), color=highlight[key]["Color"])
    #        ax_hist_x.fill_between(xs, np.zeros(len(xs)), spl(xs), color=highlight[key]["Color"], alpha=.6)
    #
    ax_hist_y = fig.add_subplot(gs[1,1])
    ax_hist_y.axis("off")
    
    hist = np.histogram(data_to_plot[key_y].values, range=y_lim, bins=n_bins, density=True)
    y = (hist[1][1:]+hist[1][:-1])/2
    
    spl = UnivariateSpline(np.insert(y,len(y),hist[1][-1]),np.insert(hist[0],len(hist[0]),hist[0][-1]))
    spl.set_smoothing_factor(smoothing_factor)
    
    ys = np.linspace(y_lim[0],y_lim[1],1000)
    
    ax_hist_y.plot(spl(ys),ys, color=keys[key_y]["Color"])
    ax_hist_y.fill_betweenx(ys,spl(ys),np.zeros(len(ys)), color=keys[key_y]["Color"], alpha=.6)
    ax_hist_y.plot([0,np.nanmax(hist[0])*1.1],[np.nanmedian(data_to_plot[key_y]),np.nanmedian(data_to_plot[key_y])],c="k",ls="--")
    
    ax_hist_y.set_ylim(y_lim)     
    ax_hist_y.set_xlim(0,np.nanmax(hist[0])*1.1)
    
    plt.savefig(outfile, bbox_inches="tight")
    
def plot_correlations_heatmap(data_frame, keys, ref_keys, outfile="Out_correlation_heatmap.pdf", corr_type="Spearman", p_values=None, alpha=0.05, v_lim=None, fs=15, cmap = "seismic"):
    if corr_type == "Spearman":
        data_to_plot = np.asarray([[spearmanr(data_frame.copy()[[key_1,key_2]].dropna()[key_1],data_frame.copy()[[key_1,key_2]].dropna()[key_2])[0] for key_1 in keys] for key_2 in ref_keys])
        if not p_values:
            p_values = np.asarray([[spearmanr(data_frame.copy()[[key_1,key_2]].dropna()[key_1],data_frame.copy()[[key_1,key_2]].dropna()[key_2])[1] for key_1 in keys] for key_2 in ref_keys])
            p_adjusted = multipletests(p_values.reshape(-1),alpha=alpha,method="fdr_bh")[1].reshape(np.shape(p_values))
        else:
            p_adjusted = p_values
    elif corr_type == "Pearson":
        data_to_plot = np.asarray([[pearsonr(data_frame.copy()[[key_1,key_2]].dropna()[key_1],data_frame.copy()[[key_1,key_2]].dropna()[key_2])[0] for key_1 in keys] for key_2 in ref_keys])
        if not p_values:
            p_values = np.asarray([[pearsonr(data_frame.copy()[[key_1,key_2]].dropna()[key_1],data_frame.copy()[[key_1,key_2]].dropna()[key_2])[1] for key_1 in keys] for key_2 in ref_keys])
            p_adjusted = multipletests(p_values.reshape(-1),alpha=alpha,method="fdr_bh")[1].reshape(np.shape(p_values))
        else:
            p_adjusted = p_values
    else:
         raise ValueError("corr_type not found")   
    
    if not v_lim:
        v_lim = (-np.ceil(10*np.nanmax(np.abs(data_to_plot)))/10,np.ceil(10*np.nanmax(np.abs(data_to_plot)))/10)
    
    #fig = plt.figure()
    fig, ax = plt.subplots()
    fig.set_size_inches((len(keys)/2)+1,(len(ref_keys)/2)+1)
    #gs = gridspec.GridSpec(2, 1)
    
    #ax = fig.add_subplot(gs[0,0])
    im = ax.matshow(data_to_plot, cmap=cmap, vmin=v_lim[0], vmax=v_lim[1])
    
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([keys[key]["Label"] for key in keys],rotation=90, fontsize=fs)
    
    ax.set_yticks(np.arange(len(ref_keys)))
    ax.set_yticklabels([ref_keys[key]["Label"] for key in ref_keys], fontsize=fs)
    
    ax.tick_params(axis="both", labelsize=fs)
    ax.tick_params(axis="x", bottom=False)
    
    for ind in range(len(keys)):
        for ind2 in range(len(ref_keys)):
            if p_adjusted[ind2,ind]<alpha:
                ax.plot(ind,ind2, c="k", marker=(8, 2, 0))
            
    cbar = plt.colorbar(im, orientation="horizontal")
    cbar.set_label(corr_type+" correlation", fontsize=fs)
    cbar.ax.tick_params(axis="x", labelsize=fs)
    
    plt.tight_layout()
    plt.savefig(outfile,bbox_inches="tight") 
    
def plot_correlations_heatmap_selection(data_frame, keys, selections, ref_key, outfile="Out_correlation_heatmap_selection.pdf", corr_type="Spearman", p_values=None, alpha=0.05, v_lim=None, fs=15, cmap = "seismic"):
    # include kendall-tau
    if corr_type == "Spearman":
        data_to_plot = np.asarray([[spearmanr(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[0] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column)] for key in keys])
        if not p_values:
            p_values = np.asarray([[spearmanr(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[1] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column)] for key in keys])
            p_adjusted = multipletests(p_values.reshape(-1),alpha=alpha,method="fdr_bh")[1].reshape(np.shape(p_values))
        else:
            p_adjusted = p_values
    elif corr_type == "Pearson":
        data_to_plot = np.asarray([[pearsonr(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[0] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column)] for key in keys])
        if not p_values:
            p_values = np.asarray([[pearsonr(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[1] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column)] for key in keys])
            p_adjusted = multipletests(p_values.reshape(-1),alpha=alpha,method="fdr_bh")[1].reshape(np.shape(p_values))
        else:
            p_adjusted = p_values
    else:
         raise ValueError("corr_type not found") 
         
    if not v_lim:
        v_lim = (-np.ceil(10*np.nanmax(np.abs(data_to_plot)))/10,np.ceil(10*np.nanmax(np.abs(data_to_plot)))/10)
    
    fig, ax = plt.subplots()
    fig.set_size_inches((len(selections)/2)+1,(len(keys)/2)+1)
    
    im = ax.matshow(data_to_plot, cmap=cmap, vmin=v_lim[0], vmax=v_lim[1])
    
    ax.set_xticks(np.arange(len(selections)))
    ax.set_xticklabels([selections[selection] for selection in selections],rotation=90, fontsize=fs)
    
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([keys[key]["Label"] for key in keys], fontsize=fs)
    
    ax.tick_params(axis="both", labelsize=fs)
    ax.tick_params(axis="x", bottom=False)
    
    for ind in range(len(selections)):
        for ind2 in range(len(keys)):
            if p_adjusted[ind2,ind]<alpha:
                ax.plot(ind,ind2, c="k", marker=(8, 2, 0))
                
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    
    #im_ratio = data_to_plot.shape[0]/data_to_plot.shape[1]
    #cbar = plt.colorbar(im, fraction=0.046*im_ratio, pad=0.04, orientation="vertical")
    
    cbar = plt.colorbar(im, orientation="vertical", cax=cax)
    cbar.set_label(corr_type+" correlation", fontsize=fs)
    cbar.ax.tick_params(axis="y", labelsize=fs)
    
    plt.tight_layout()
    plt.savefig(outfile,bbox_inches="tight") 
    
def plot_correlations_heatmap_selection_double(data_frame, keys, selections, key_2, outfile="Out_correlation_heatmap_selection.pdf", corr_type="Spearman", p_values=None, alpha=0.05, v_lim=None, fs=15, cmap = "seismic"):
    
    if corr_type == "Spearman":
        data_to_plot = np.asarray([[spearmanr(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[0] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column) for ref_key in data_frame.columns if re.search(selection,ref_key) and re.search(key_2,ref_key)] for key in keys])
        if not p_values:
            p_values = np.asarray([[spearmanr(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[1] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column) for ref_key in data_frame.columns if re.search(selection,ref_key) and re.search(key_2,ref_key)] for key in keys])
            p_adjusted = multipletests(p_values.reshape(-1),alpha=alpha,method="fdr_bh")[1].reshape(np.shape(p_values))
        else:
            p_adjusted = p_values
    elif corr_type == "Pearson":
        data_to_plot = np.asarray([[pearsonr(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[0] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column) for ref_key in data_frame.columns if re.search(selection,ref_key) and re.search(key_2,ref_key)] for key in keys])
        if not p_values:
            p_values = np.asarray([[pearsonr(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[1] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column) for ref_key in data_frame.columns if re.search(selection,ref_key) and re.search(key_2,ref_key)] for key in keys])
            p_adjusted = multipletests(p_values.reshape(-1),alpha=alpha,method="fdr_bh")[1].reshape(np.shape(p_values))
        else:
            p_adjusted = p_values
    else:
         raise ValueError("corr_type not found") 
         
    if not v_lim:
        v_lim = (-np.ceil(10*np.nanmax(np.abs(data_to_plot)))/10,np.ceil(10*np.nanmax(np.abs(data_to_plot)))/10)
    
    fig, ax = plt.subplots()
    fig.set_size_inches((len(selections)/2)+1,(len(keys)/2)+1)
    
    im = ax.matshow(data_to_plot, cmap=cmap, vmin=v_lim[0], vmax=v_lim[1])
    
    ax.set_xticks(np.arange(len(selections)))
    ax.set_xticklabels([selections[selection] for selection in selections],rotation=90, fontsize=fs)
    
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([keys[key]["Label"] for key in keys], fontsize=fs)
    
    ax.tick_params(axis="both", labelsize=fs)
    ax.tick_params(axis="x", bottom=False)
    
    for ind in range(len(selections)):
        for ind2 in range(len(keys)):
            if p_adjusted[ind2,ind]<alpha:
                ax.plot(ind,ind2, c="k", marker=(8, 2, 0))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    
    cbar = plt.colorbar(im, orientation="vertical", cax=cax)
    cbar.set_label(corr_type+" correlation", fontsize=fs)
    cbar.ax.tick_params(axis="y", labelsize=fs)
    
    plt.tight_layout()
    plt.savefig(outfile,bbox_inches="tight") 
    
def plot_correlations_grid(data_frame, keys, ref_key, outfile = "Out_correlation_grid,pdf", y_label = "y", n_columns = 4, fs = 15, n_rows = None, x_lim = None, y_lim = None, plot_linreg = True, legend_loc = "upper left"):
    
    if not n_rows:
        n_rows = int(np.ceil(len(keys)/n_columns))
        
    if n_columns * n_rows < len(keys):
        raise ValueError("Grid resolution not fitting a plots. Increase n_columns or n_rows.")
    
    key_x = list(ref_key.keys())[0]
    
    fig = plt.figure()
    fig.set_size_inches((n_columns*3)+1.2,(n_rows*3)+0.9)
    gs = gridspec.GridSpec(n_rows+1, n_columns+1, wspace=0.03, hspace=0.03, width_ratios=[4]+[10]*n_columns, height_ratios=[3]+[10]*n_rows)
    
    if not x_lim:
        x_lim = (np.nanmin(data_frame[key_x])-(0.05*np.nanmax(data_frame[key_x])),np.nanmax(data_frame[key_x])+(0.05*np.nanmax(data_frame[key_x])))  
    
    if not y_lim:
        y_lim = (np.nanmin(data_frame[list(keys.keys())])-(0.05*np.nanmax(data_frame[list(keys.keys())])),np.nanmax(data_frame[list(keys.keys())])+(0.05*np.nanmax(data_frame[list(keys.keys())])))
          
    for ind,key_y in enumerate(keys):
        data_to_plot = data_frame.copy()[[key_x,key_y]]
        
        ax = fig.add_subplot(gs[int(np.floor(ind/n_columns))+1,int(np.mod(ind,n_columns))+1])        
        ax.scatter(data_to_plot[key_x],data_to_plot[key_y],facecolors="None",color=keys[key_y]["Color"], marker=".",s=40, label=keys[key_y]["Label"])
            
        if plot_linreg:
            coef = np.polyfit(data_to_plot[key_x],data_to_plot[key_y],1)
            poly1d_fn = np.poly1d(coef)
            ax.plot(x_lim,poly1d_fn(x_lim),c="k")
            
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
        ax.tick_params(axis="y", labelsize=fs)
        ax.tick_params(axis="x", labelsize=fs)
        ax.xaxis.set_ticks_position('top')
        
        if not (int(np.mod(ind,n_columns)) == 0):
            ax.set_yticklabels([])
        if not (int(np.floor(ind/n_columns)) == 0):
            ax.set_xticklabels([])
        
        ax.grid(axis='both', color='0.8')
        
        try:
            legend = ax.legend(loc=legend_loc, fontsize=fs)
        except:
            print("legend_loc not found. Using upper left as a default.")
            legend = ax.legend(loc="upper left", fontsize=fs, shadow=True, fancybox=True)
            
        legend.get_frame().set_linewidth(2)

        for legobj in legend.legendHandles:
            legobj.set_sizes([100])
            legobj.set_linewidth(5)
            
    ax_x = fig.add_subplot(gs[0,1:])
    ax_x.set_xlim(0,1)
    ax_x.set_ylim(0,1)
    ax_x.axis('off')
    
    ax_x.text(0.5,1.0,ref_key[key_x]["Label"], fontsize=fs, horizontalalignment='center', verticalalignment='top')
    
    ax_y = fig.add_subplot(gs[1:,0])
    ax_y.set_xlim(0,1)
    ax_y.set_ylim(0,1)
    ax_y.axis('off')
  
    ax_y.text(0.0,0.5,y_label, fontsize=fs, rotation=90, horizontalalignment='left', verticalalignment='center')
    
    plt.savefig(outfile, bbox_inches="tight")

def plot_correlations_boxplot(data_frame, keys, ref_key, ind_key=0, selection=None, corr_type="Spearman", outfile="Out_boxplot.pdf", fs=15, y_lim=None, grid=True):

    key_x = list(ref_key.keys())[ind_key]
    
    if selection:
        if corr_type=="Spearman":
            collection_R = [[spearmanr(data_frame.copy()[[key_x,column]].dropna()[key_x],data_frame.copy()[[key_x,column]].dropna()[column])[0] for column in data_frame.columns if re.findall(key,column) for el in selection if re.findall(el,column)] for key in keys]
        elif corr_type=="Pearson":
            collection_R = [[pearsonr(data_frame.copy()[[key_x,column]].dropna()[key_x],data_frame.copy()[[key_x,column]].dropna()[column])[0] for column in data_frame.columns if re.findall(key,column) for el in selection if re.findall(el,column)] for key in keys]            
        else:
            raise ValueError("Correlation type not defined.")    
    else:
        if corr_type=="Spearman":
            collection_R = [[spearmanr(data_frame.copy()[[key_x,column]].dropna()[key_x],data_frame.copy()[[key_x,column]].dropna()[column])[0] for column in data_frame.columns if re.findall(key,column)] for key in keys]
        elif corr_type=="Pearson":
            collection_R = [[pearsonr(data_frame.copy()[[key_x,column]].dropna()[key_x],data_frame.copy()[[key_x,column]].dropna()[column])[0] for column in data_frame.columns if re.findall(key,column)] for key in keys]
        else:
            raise ValueError("Correlation type not defined.")
    
    
    fig,ax = plt.subplots()
    fig.set_size_inches(len(keys)/3+1,5)
    bplot = ax.boxplot(collection_R,patch_artist=True,medianprops=dict(color="k"))
    
    colors = [keys[key]["Color"] for key in keys]
    for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    
    ax.plot([0.5,20.5],[0,0],c="k",ls=":")
    
    if grid:
        ax.grid(axis='both', color='0.8')
    
    if not y_lim:
        y_lim=((-1)*np.nanmax(np.abs(collection_R))-0.05*np.nanmax(np.abs(collection_R)),np.nanmax(np.abs(collection_R))+0.05*np.nanmax(np.abs(collection_R)))
    
    ax.set_xticklabels(list(keys.keys()),rotation=90, fontsize=fs)
    ax.set_ylim(y_lim)
    ax.set_ylabel(corr_type+"R "+ref_key[key_x]["Label"], fontsize=fs)
    
    ax.tick_params(axis="y",labelsize=fs)
    
    plt.savefig(outfile, bbox_inches="tight")

# Average
def plot_binned_average(data_list, N=100, outfile="Out_binned_average", cmap = "viridis", fs = 15, x_label = "rel. Index", y_label="Mean(y)", boundaries=None, y_lim=None, grid = True):

    data_grouped = np.asarray([[np.nanmean(item[int(np.ceil((n)*len(item)/N)):int(np.ceil((n+1)*len(item)/N))]) for n in range(N)] for item in data_list])
    mean_grouped = np.nanmean(data_grouped, axis=0)
    std_grouped = np.nanstd(data_grouped, axis=0)
    cv_grouped = std_grouped/mean_grouped

    try:
        cm = mpl.cm.get_cmap(cmap)
    except:
        raise ValueError("Color map not found")
        
    min_cv = np.min(cv_grouped)
    max_cv = np.max(cv_grouped)
    
    fig = plt.figure()
    fig.set_size_inches(6,4)
    gs = gridspec.GridSpec(1,2, width_ratios=[5,1])
    
    xs = np.arange((1/(2*N)),1+(1/(2*N)),1/N)
    
    ax = fig.add_subplot(gs[0,0])
    for ind in range(N-1):
        ax.plot([xs[ind],xs[ind+1]], [mean_grouped[ind],mean_grouped[ind+1]], lw=3, c=cm((((cv_grouped[ind]+cv_grouped[ind+1])/2)-min_cv)/(max_cv-min_cv)))
        ax.fill_between([xs[ind],xs[ind+1]], [mean_grouped[ind]-std_grouped[ind],mean_grouped[ind+1]-std_grouped[ind+1]],[mean_grouped[ind]+std_grouped[ind],mean_grouped[ind+1]+std_grouped[ind+1]], color=cm((((cv_grouped[ind]+cv_grouped[ind+1])/2)-min_cv)/(max_cv-min_cv)), alpha=0.6)
    
    if boundaries:
        for boundary in boundaries:
            ax.plot([0,1],[boundary,boundary],c="k",ls="--")
            
    ax.set_xlabel(x_label, fontsize=fs)
    ax.set_ylabel(y_label, fontsize=fs)
    
    ax.set_xlim(0,1)
    if y_lim:
        ax.set_ylim(y_lim)
    
    ax.tick_params(axis="both",labelsize=fs)
    
    if grid:
        ax.grid(axis='both', color='0.8')
        
    cax = fig.add_subplot(gs[0,1])
    cax.matshow(np.linspace(min_cv,max_cv,101).reshape(-1,1), cmap=cm, aspect=0.1)
    
    cax.set_xticks([])
    
    cax.yaxis.tick_right()
    cax.set_yticks(np.linspace(0,100,6), [str(round(float(label), 2)) for label in (np.linspace(0,1,6)*(max_cv-min_cv))+min_cv], fontsize=fs)
    
    cax.set_ylabel("Coefficient of variation", fontsize=fs, rotation=90)
    
    plt.savefig(outfile, bbox_inches="tight")
        
# Enrichment    
def plot_enrichment(data_frame, keys, p_column, size_column, outfile = "../Out_enrichment.pdf", fs = 15, auc_column=None, cmap="Reds", splits=None, already_log10_transformed=False, v_lim=None, cbar_resolution = 4, num_legend_labels = 3, label=True):

    if not already_log10_transformed:
        data_frame = data_frame.copy()
        data_frame[p_column] = np.log10(data_frame[p_column])
    
    if not v_lim:
        v_lim = (np.floor(np.nanmin(data_frame[p_column])),0)
    
    dv = v_lim[1]-v_lim[0]
    try:
        cm = mpl.cm.get_cmap(cmap)
    except:
        raise ValueError("Colormap not found")
    
    fig,ax=plt.subplots()
    
    if auc_column:
        fig.set_size_inches(2.5,len(keys)/2)
    else:
        fig.set_size_inches(0.5,len(keys)/2)
    
    sc = ax.scatter(np.ones(len(keys))*(-1),np.ones(len(keys))*(-1),s=data_frame.loc[list(keys.keys()),size_column],color="k", alpha=0.5)
    
    for ind,index in enumerate(keys.keys()):
        ax.scatter(1,ind,s=data_frame.loc[index,size_column],color=cm(-1*data_frame.loc[index,p_column]/dv))
        if auc_column:
            ax.text(1.012,ind,"AUC = {:.2f}".format(data_frame.loc[index,auc_column]),verticalalignment="center",fontsize=fs)
            
    ax.set_yticks(np.arange(0,len(keys)))
    ax.set_yticklabels([item[1] for item in keys.items()], fontsize=fs)
    
    ax.set_xticks([])
    ax.set_ylim(len(keys)-0.5,-0.5)
    if auc_column:
        ax.set_xlim(0.99,1.08)
    else:
        ax.set_xlim(0.99,1.01)
        
    if splits:
        for split in splits:
            ax.plot([0.99,1.08],[split+0.5,split+0.5],c="k",ls=":")
    
    if len(keys)<10:
        # Maybe add dynamic shrinkage?
        if auc_column:
            c_map_ax = fig.add_axes([1.0, 0.15, 0.1, 0.7])
        else:
            c_map_ax = fig.add_axes([1.3, 0.15, 0.45, 0.7])
    else:
        if auc_column:
            c_map_ax = fig.add_axes([1.0, 0.3, 0.1, 0.4])
        else:
            c_map_ax = fig.add_axes([1.3, 0.3, 0.45, 0.4])
    mpl.colorbar.ColorbarBase(c_map_ax, cmap=cmap, orientation = 'vertical')
    if label:
        c_map_ax.set_ylabel("log10(adj. p-value)",rotation=90, fontsize=fs, labelpad=10)
    
    c_map_ax.set_yticks(np.arange(v_lim[1],v_lim[0]-0.01,np.floor(v_lim[0]/cbar_resolution))/dv*(-1))
    c_map_ax.set_yticklabels(np.arange(v_lim[1],v_lim[0]-0.01,np.floor(v_lim[0]/cbar_resolution)),fontsize=fs)
    
    if len(keys)==1:
        ax.legend(*sc.legend_elements("sizes", num=num_legend_labels),fontsize=fs, ncol=1, bbox_to_anchor=(0.85, 0))
    else:
        ax.legend(*sc.legend_elements("sizes", num=num_legend_labels),fontsize=fs, ncol=2, bbox_to_anchor=(1.15, 0))
    plt.savefig(outfile,bbox_inches="tight")
       
def plot_AUC(data_frame_keys, keys, data_frame_ref_keys, ref_keys, outfile="Out_AUC.pdf", fs=15, grid=True, fs_legend=15):
        
    indices = [index for index in data_frame_ref_keys.index if index in data_frame_keys.index]
    
    for ind,ref_key in enumerate(ref_keys):
        
        fig,ax = plt.subplots()
        fig.set_size_inches(5,5)
        
        for key in keys:        
            fpr, tpr, _ = roc_curve(data_frame_keys.loc[indices,key],data_frame_ref_keys.loc[indices,ref_key])
            ax.plot(fpr, tpr, lw=5, label= keys[key]["Label"]+": {:.2f}".format(roc_auc_score(data_frame_keys.loc[indices,key],data_frame_ref_keys.loc[indices,ref_key])),c=keys[key]["Color"],zorder=20)
    
        # Plot reference line
        ax.plot([0,1],[0,1],ls="--",c="k",lw=3)
        
        # Set layout
        ax.set_xlabel("1-Specificity",fontsize=fs)
        ax.set_ylabel("Sensitivity",fontsize=fs)
        ax.set_xticks([0.25,0.5,0.75,1.0])
        ax.set_yticks([0.0,0.25,0.5,0.75,1.0])
        ax.set_xticklabels([0.25,0.5,0.75,1.0],fontsize=fs)
        ax.set_yticklabels([0.0,0.25,0.5,0.75,1.0],fontsize=fs)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        plt.legend(loc="lower right",fontsize=fs_legend, shadow=True, fancybox=True, framealpha=1)
        
        if grid:
            ax.grid(axis='both', color='0.8')
        
        if len(ref_keys)>1:
            plt.savefig(outfile[:-4]+"_"+str(ind)+outfile[-4:],bbox_inches="tight")  
        else:
            plt.savefig(outfile,bbox_inches="tight")  
            
        
        