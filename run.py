#!/usr/bin/python

#example script for training and test

import os
import glob
import shutil
import subprocess
import drawVis
import re
import numpy as np

def train(dataset, aes, jaes, ind):
    """train image sae & text sae simultaneously; train msae"""
    print '###########Training#################'
    #output_dir=os.path.join('data/',dataset,'output','output'+str(ind))
    config=os.path.join("config/",dataset,"config"+str(ind)+".ini")
    
    for m in xrange(len(aes)):
        cmd1="python main.py -a "
        for ae in aes[m]:
            cmd1+=ae+" "
        cmd1+=config
        print cmd1
        isae=subprocess.Popen(cmd1, shell=True)
        isae.wait()
      
    cmd3="python main.py -a "
    for ae in jaes:
        cmd3+=ae+" "
    cmd3+=config
    print cmd3
    jsae=subprocess.Popen(cmd3, shell=True)
    jsae.wait()

    cmd4="python main.py -a msae "+config
    print cmd4
    subprocess.call(cmd4,shell=True)
    
if __name__=='__main__':
    """to run on other datasets, just update the following information"""

    datasets = ['tiny']
    config_cnt = 1
    modalscnt = 9
    aes = [[] for x in xrange(modalscnt)]
    
    aes[0]=['colorae125-128', 'colorae128-64', 'colorae64-32', 'colorae32-16', 'colorae16-8', 'colorae8-4', 'colorsae128']
    aes[1]=['lbpae256-128', 'lbpae128-64', 'lbpae64-32', 'lbpae32-16', 'lbpae16-8', 'lbpae8-4', 'lbpsae128']
    aes[2]=['gistae512-128', 'gistae128-64', 'gistae64-32', 'gistae32-16', 'gistae16-8', 'gistae8-4', 'gistsae128']
    aes[3]=['gaborae512-128', 'gaborae128-64', 'gaborae64-32', 'gaborae32-16', 'gaborae16-8', 'gaborae8-4', 'gaborsae128']
    aes[4]=['edgeae512-128', 'edgeae128-64', 'edgeae64-32', 'edgeae32-16', 'edgeae16-8', 'edgeae8-4', 'edgesae128']
    aes[5]=['sift200ae1000-128', 'sift200ae128-64', 'sift200ae64-32', 'sift200ae32-16', 'sift200ae16-8', 'sift200ae8-4', 'sift200sae128']
    aes[6]=['sift1000ae1000-128', 'sift1000ae128-64', 'sift1000ae64-32', 'sift1000ae32-16', 'sift1000ae16-8', 'sift1000ae8-4', 'sift1000sae128']
    aes[7]=['surf200ae1000-128', 'surf200ae128-64', 'surf200ae64-32', 'surf200ae32-16', 'surf200ae16-8', 'surf200ae8-4', 'surf200sae128']
    aes[8]=['surf1000ae1000-128', 'surf1000ae128-64', 'surf1000ae64-32', 'surf1000ae32-16', 'surf1000ae16-8', 'surf1000ae8-4', 'surf1000sae128']
    jaes=['jae256-64', 'jsae64']

    for ds in datasets:
        for ind in [x for x in range(config_cnt)]:
            train(ds, aes, jaes, ind)
