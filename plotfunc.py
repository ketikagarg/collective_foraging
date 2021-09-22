#!/usr/bin/env python
# coding: utf-8

# In[15]:


from __future__ import division, print_function

import random
import math
import os
import random
import shutil
import numpy as np
from scipy import spatial as st 
import matplotlib.pyplot as plt

import matplotlib.ticker as ticker
from periodic_kdtree import PeriodicCKDTree
from sklearn.cluster import DBSCAN
from scipy import stats as sts
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pylab as pl

import sys
import copy
import itertools
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import pylab as pl
from IPython import display
from IPython.display import display, clear_output
# from ipynb.fs.full.main_module import main
from main_module import main


# In[1]:


folder = "data"
radius_interaction = [1]




# In[30]:


def draw(w):
    
#     from main_module import main
    fig = plt.figure(figsize=(9,9))
    ax = fig.subplots(1,1)

    plt.ion()
    ax.set_facecolor('white')
    fig.show()
    fig.canvas.draw()

    trees = [w.kwargs['targets']]
    betas = [w.kwargs['beta']]
    agents = [w.kwargs['agents']]
    mu_range = [w.kwargs['mu']]
    alpha_range = [float(w.kwargs['alpha'])]
    tick_speed = w.kwargs['tick_speed']
    runs = float(w.kwargs['runs'])
    save = w.kwargs['save']
#     print(save)
#     runs = 10
 

    var_list = [trees[0], agents[0], betas[0], mu_range[0], alpha_range[0]]
    string_ints = [str(int) for int in var_list]
    z = 0
    filename = "_".join(string_ints)
    if save == True:
        if os.path.exists(folder) and os.path.isdir(folder):
            shutil.rmtree(folder)

        os.mkdir(folder)
#         filename = "test"
        f2 = open(folder+"/"+filename+".txt", 'a')
        
    while z < runs:     
        paramlist = list(itertools.product(trees,agents,betas,radius_interaction, mu_range, alpha_range))
        print(paramlist[0])
        result = main(paramlist[0], fig, ax, tick_speed)
        if save == True:
            f2.write(result)
        z = z+1
    if save == True:
        f2.close()
            


# In[2]:


def f(targets, beta, agents, mu, alpha, tick_speed, runs, save):
    return

def create_widgets() :
    w = interactive(f, targets = widgets.IntSlider(min=0, max=10000, step=1000, value=1000), beta = widgets.FloatSlider(min=1.1, max=3.5, step=0.1, value=3), agents = widgets.IntSlider(min=1, max=100, step=1, value=10), mu = widgets.FloatSlider(min=1.1, max=3.5, step=0.1, value=3), alpha = widgets.Text('0.001'), tick_speed = widgets.IntSlider(min=1, max=100, step=1, value=10),  runs = widgets.Text('10'), save=widgets.Checkbox(value=False, description='Save output?')  )
    return w


# In[ ]:





# In[ ]:





# In[ ]:




