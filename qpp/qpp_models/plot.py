import numpy as np
import os, sys
import pandas as pd
import matplotlib.pyplot as plt

def bxplot_w_info(data, y_label):
    plt.clf()
   
    fig, ax = plt.subplots()
    """boxes = [
        {
            'label' : "Male height",
            'whislo': 162.6,    # Bottom whisker position
            'q1'    : 170.2,    # First quartile (25th percentile)
            'med'   : 175.7,    # Median         (50th percentile)
            'q3'    : 180.4,    # Third quartile (75th percentile)
            'whishi': 187.8,    # Top whisker position
            'fliers': []        # Outliers
        }
    
    ]"""
    plt.yscale('log')
    ax.bxp(data, showfliers=False)
    ax.set_ylabel(y_label)
    
    #plt.savefig("boxplot.png")
    #plt.close()
    plt.show()
    
# The data dictionary should contain mapping from columns to specific values.
def plot_bar_specific(data:dict):
    plt.clf()
    
    plt.bar(range(len(data)), list(data.values()), align='center')
    plt.xticks(range(len(data)), list(data.keys()))
    plt.show()
if __name__ == '__main__':
    pass